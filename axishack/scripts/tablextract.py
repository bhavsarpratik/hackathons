import os
import re
from math import ceil, degrees, floor, radians

import cv2
import numpy as np
from pdf2image import convert_from_path
from pdftabextract import imgproc
from pdftabextract.clustering import (calc_cluster_centers_1d,
                                      find_clusters_1d_break_dist,
                                      zip_clusters_and_values)
from pdftabextract.common import (DIRECTION_VERTICAL, ROTATION, SKEW_X, SKEW_Y,
                                  all_a_in_b, parse_pages, read_xml,
                                  save_page_grids)
from pdftabextract.extract import (datatable_to_dataframe, fit_texts_into_grid,
                                   make_grid_from_positions)
from pdftabextract.geom import pt
from pdftabextract.textboxes import (border_positions_from_texts,
                                     deskew_textboxes, join_texts,
                                     rotate_textboxes,
                                     split_texts_by_positions)
from tabula import read_pdf

import camelot


def do_tablextract(self, g, pdf_path, p_num):  # g is globals
    print('Starting tablextract')
    camelot_method = 'lattice' #stream/lattice

    if self.pdf_type == 'normal':
        print(pdf_path, p_num)
        if 'tabula' in g.text_pdf_method:
            tables = read_pdf(pdf_path, pages=[p_num], multiple_tables=True, java_options='-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider')
            for i in range(len(tables)):
                table_file_path = '%s/%s-%s' % (self.tables_folder_tabula, p_num, i)
                # tables[i].fillna('').to_html('%s.html' % (table_file_path))
                try:
                    tables[i].fillna('').to_csv('%s.csv' % (table_file_path), encoding='utf-8')
                except:
                    tables[i].fillna('').to_csv('%s.csv' % (table_file_path), encoding='cp1252')
        if 'camelot' in g.text_pdf_method:
            tables = camelot.read_pdf(pdf_path,  flavor=camelot_method, pages=str(p_num))
            for i in range(len(tables)):
                # print(tables[0].parsing_report)
                table_file_path = '%s/%s-%s.csv' % (self.tables_folder_camelot, p_num, i)
                tables.export(table_file_path, f='csv', compress=False)

    else:
        # trying camelot
        print('Doing camelot-stream')
        camelot_method = 'stream' #stream/lattice
        tables = camelot.read_pdf(pdf_path,  flavor=camelot_method, pages=str(p_num))
        for i in range(len(tables)):
            # print(tables[0].parsing_report)
            table_file_path = '%s/%s-%s.csv' % (self.tables_folder_camelot, p_num, i)
            tables.export(table_file_path, f='csv', compress=False)

        # Trying pdftabextract
        filename = os.path.basename(pdf_path).split('.')[0].split('/')[0]
        DATAPATH = self.images_folder  # 'data/'
        INPUT_XML = '%s/%s.xml' % (self.images_folder, filename)
        os.system("pdftohtml -c -hidden -xml -enc UTF-8  -f %s -l %s %s %s" % (p_num, p_num, pdf_path, INPUT_XML))
        # os.system("pdftohtml -c -hidden -f %s -l %s %s %s/%s.html" % (p_num, p_num, pdf_path, self.html_folder, filename))

        # Load the XML that was generated with pdftohtml
        xmltree, xmlroot = read_xml(INPUT_XML)
        # parse it and generate a dict of pages
        pages = parse_pages(xmlroot)
        # print(pages[p_num]['texts'][0])
        p = pages[p_num]

        # Detecting lines
        if self.doc_type == 'image':
            imgfilebasename = '%s-%s_1' % (filename, p_num)
            imgfile = self.file_path
        elif self.doc_type == 'pdf':
            try:
                imgfilebasename = '%s-%s_1' % (filename, p_num)
                imgfile = '%s/%s-%s_1.png' % (DATAPATH, filename, p_num)
            except:
                imgfilebasename = filename + str(p_num)
                imgfile = '%s/%s-%s_1.png' % (DATAPATH, filename, p_num)

        print("\npage %d: detecting lines in image file '%s'..." % (p_num, imgfile))

        # create an image processing object with the scanned page
        iproc_obj = imgproc.ImageProc(imgfile)

        # calculate the scaling of the image file in relation to the text boxes coordinate system dimensions
        page_scaling_x = iproc_obj.img_w / p['width']   # scaling in X-direction
        page_scaling_y = iproc_obj.img_h / p['height']  # scaling in Y-direction

        # detect the lines
        lines_hough = iproc_obj.detect_lines(canny_kernel_size=3, canny_low_thresh=50, canny_high_thresh=150,
                                             hough_rho_res=1,
                                             hough_theta_res=np.pi / 500,
                                             hough_votes_thresh=round(0.2 * iproc_obj.img_w))
        print("> found %d lines" % len(lines_hough))

        # helper function to save an image
        def save_image_w_lines(iproc_obj, imgfilebasename):
            img_lines = iproc_obj.draw_lines(orig_img_as_background=True)
            img_lines_file = os.path.join(self.temp_folder, '%s-lines-orig.png' % imgfilebasename)

            print("> saving image with detected lines to '%s'" % img_lines_file)
            cv2.imwrite(img_lines_file, img_lines)

        save_image_w_lines(iproc_obj, imgfilebasename)

        # find rotation or skew
        # the parameters are:
        # 1. the minimum threshold in radians for a rotation to be counted as such
        # 2. the maximum threshold for the difference between horizontal and vertical line rotation (to detect skew)
        # 3. an optional threshold to filter out "stray" lines whose angle is too far apart from the median angle of
        #    all other lines that go in the same direction (no effect here)
        rot_or_skew_type, rot_or_skew_radians = iproc_obj.find_rotation_or_skew(radians(0.5),    # uses "lines_hough"
                                                                                radians(1),
                                                                                omit_on_rot_thresh=radians(0.5))

        # rotate back or deskew text boxes
        needs_fix = True
        if rot_or_skew_type == ROTATION:
            print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
            rotate_textboxes(p, -rot_or_skew_radians, pt(0, 0))
        elif rot_or_skew_type in (SKEW_X, SKEW_Y):
            print("> deskewing in direction '%s' by %f°" % (rot_or_skew_type, -degrees(rot_or_skew_radians)))
            deskew_textboxes(p, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
        else:
            needs_fix = False
            print("> no page rotation / skew found")

        if needs_fix:
            # rotate back or deskew detected lines
            lines_hough = iproc_obj.apply_found_rotation_or_skew(rot_or_skew_type, -rot_or_skew_radians)

            save_image_w_lines(iproc_obj, imgfilebasename + '-repaired')

        # save repaired XML (i.e. XML with deskewed textbox positions)

        repaired_xmlfile = os.path.join(self.temp_folder, filename + '.repaired.xml')

        print("saving repaired XML file to '%s'..." % repaired_xmlfile)
        xmltree.write(repaired_xmlfile)

        # Clustering vertical lines
        # cluster the detected *vertical* lines using find_clusters_1d_break_dist as simple clustering function
        # (break on distance MIN_COL_WIDTH/2)
        # additionally, remove all cluster sections that are considered empty
        # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
        # per cluster section
        MIN_COL_WIDTH = g.MIN_COL_WIDTH  # minimum width of a column in pixels, measured in the scanned pages
        vertical_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_VERTICAL, find_clusters_1d_break_dist,
                                                    remove_empty_cluster_sections_use_texts=p['texts'],  # use this page's textboxes
                                                    remove_empty_cluster_sections_n_texts_ratio=0.1,    # 10% rule
                                                    remove_empty_cluster_sections_scaling=page_scaling_x,  # the positions are in "scanned image space" -> we scale them to "text box space"
                                                    dist_thresh=MIN_COL_WIDTH / 2)
        print("> found %d clusters" % len(vertical_clusters))

        # draw the clusters
        img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
        save_img_file = os.path.join(self.temp_folder, '%s-vertical-clusters.png' % imgfilebasename)
        print("> saving image with detected vertical clusters to '%s'" % save_img_file)
        cv2.imwrite(save_img_file, img_w_clusters)

        # Clustering horizontal lines
        # cluster the detected *horizontal* lines using find_clusters_1d_break_dist as simple clustering function
        # (break on distance MIN_ROW_WIDTH/2)
        # additionally, remove all cluster sections that are considered empty
        # a cluster is considered empty when the number of text boxes in it is below 10% of the median number of text boxes
        # per cluster section
        MIN_ROW_WIDTH = g.MIN_ROW_WIDTH  # minimum width of a row in pixels, measured in the scanned pages
        horizontal_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_HORIZONTAL, find_clusters_1d_break_dist,
                                                      remove_empty_cluster_sections_use_texts=p['texts'],  # use this page's textboxes
                                                      remove_empty_cluster_sections_n_texts_ratio=0.1,    # 10% rule
                                                      remove_empty_cluster_sections_scaling=page_scaling_y,  # the positions are in "scanned image space" -> we scale them to "text box space"
                                                      dist_thresh=MIN_ROW_WIDTH / 2)
        print("> found %d clusters" % len(horizontal_clusters))

        # draw the clusters
        img_w_clusters_hoz = iproc_obj.draw_line_clusters(imgproc.DIRECTION_HORIZONTAL, horizontal_clusters)
        save_img_file = os.path.join(self.temp_folder, '%s-horizontal-clusters.png' % imgfilebasename)
        print("> saving image with detected vertical clusters to '%s'" % save_img_file)
        cv2.imwrite(save_img_file, img_w_clusters_hoz)

        page_colpos = np.array(calc_cluster_centers_1d(vertical_clusters)) / page_scaling_x
        print('found %d column borders:' % len(page_colpos))
        print(page_colpos)

        page_rowpos = np.array(calc_cluster_centers_1d(horizontal_clusters)) / page_scaling_y
        print('found %d row borders:' % len(page_rowpos))
        print(page_rowpos)

        # right border of the second column
        col2_rightborder = page_colpos[2]

        # calculate median text box height
        median_text_height = np.median([t['height'] for t in p['texts']])

        # get all texts in the first two columns with a "usual" textbox height
        # we will only use these text boxes in order to determine the line positions because they are more "stable"
        # otherwise, especially the right side of the column header can lead to problems detecting the first table row
        text_height_deviation_thresh = median_text_height / 2
        texts_cols_1_2 = [t for t in p['texts']
                          if t['right'] <= col2_rightborder
                          and abs(t['height'] - median_text_height) <= text_height_deviation_thresh]

        # get all textboxes' top and bottom border positions
        borders_y = border_positions_from_texts(texts_cols_1_2, DIRECTION_VERTICAL)

        # break into clusters using half of the median text height as break distance
        clusters_y = find_clusters_1d_break_dist(borders_y, dist_thresh=median_text_height / 2)
        clusters_w_vals = zip_clusters_and_values(clusters_y, borders_y)

        # for each cluster, calculate the median as center
        pos_y = calc_cluster_centers_1d(clusters_w_vals)
        pos_y.append(p['height'])

        print('number of line positions:', len(pos_y))

        pttrn_table_row_beginning = re.compile(r'^[\d Oo][\d Oo]{2,} +[A-ZÄÖÜ]')

        # 1. try to find the top row of the table
        texts_cols_1_2_per_line = split_texts_by_positions(texts_cols_1_2, pos_y, DIRECTION_VERTICAL, alignment='middle', enrich_with_positions=True)

        # go through the texts line per line
        for line_texts, (line_top, line_bottom) in texts_cols_1_2_per_line:
            line_str = join_texts(line_texts)
            if pttrn_table_row_beginning.match(line_str):  # check if the line content matches the given pattern
                top_y = line_top
                break
        else:
            top_y = 0

        print('Top_y: %s' % top_y)

        # hints for a footer text box
        words_in_footer = ('anzeige', 'annahme', 'ala')

        # 2. try to find the bottom row of the table
        min_footer_text_height = median_text_height * 1.5
        min_footer_y_pos = p['height'] * 0.7
        # get all texts in the lower 30% of the page that have are at least 50% bigger than the median textbox height
        bottom_texts = [t for t in p['texts']
                        if t['top'] >= min_footer_y_pos and t['height'] >= min_footer_text_height]
        bottom_texts_per_line = split_texts_by_positions(bottom_texts,
                                                         pos_y + [p['height']],   # always down to the end of the page
                                                         DIRECTION_VERTICAL,
                                                         alignment='middle',
                                                         enrich_with_positions=True)
        # go through the texts at the bottom line per line
        page_span = page_colpos[-1] - page_colpos[0]
        min_footer_text_width = page_span * 0.8
        for line_texts, (line_top, line_bottom) in bottom_texts_per_line:
            line_str = join_texts(line_texts)
            has_wide_footer_text = any(t['width'] >= min_footer_text_width for t in line_texts)
            # check if there's at least one wide text or if all of the required words for a footer match
            if has_wide_footer_text or all_a_in_b(words_in_footer, line_str):
                bottom_y = line_top
                break
        else:
            bottom_y = p['height']

        print(bottom_y)
        print(pos_y)

        # finally filter the line positions so that only the lines between the table top and bottom are left
        print(page_rowpos)
        print("> page %d: %d lines between [%f, %f]" % (p_num, len(page_rowpos), top_y, bottom_y))

        def subsequent_pairs(l):
            """
            Return subsequent pairs of values in a list <l>, i.e. [(x1, x2), (x2, x3), (x3, x4), .. (xn-1, xn)] for a
            list [x1 .. xn]
            """
            return [(l[i - 1], v) for i, v in enumerate(l) if i > 0]

        # page_rowpos = [y for y in pos_y if top_y <= y <= bottom_y]
        print(page_colpos, page_rowpos)
        grid = make_grid_from_positions(page_colpos, page_rowpos)
        # print(grid)
        n_rows = len(grid)
        n_cols = len(grid[0])
        print("> page %d: grid with %d rows, %d columns" % (p_num, n_rows, n_cols))

        page_grids_file = os.path.join(self.temp_folder, filename + '_pagegrids.json')
        print("saving page grids JSON file to '%s'" % page_grids_file)
        save_page_grids({p_num: grid}, page_grids_file)

        datatable = fit_texts_into_grid(p['texts'], grid)
        df = datatable_to_dataframe(datatable)
        # print(df.head(n=2))

        csv_output_file = os.path.join(self.tables_folder, filename + '.csv')
        print("saving extracted data to '%s'" % csv_output_file)
        df.to_csv(csv_output_file, index=False, header=False)

        # html_output_file = os.path.join(self.tables_folder, filename + '.html')
        # print("saving extracted data to '%s'" % html_output_file)
        # df.to_html(html_output_file, index=False, header=False)
