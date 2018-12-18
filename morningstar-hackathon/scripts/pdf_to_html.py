import glob
import json
import os
import pickle
import re
import shutil
from math import degrees, radians
from multiprocessing.dummy import Pool

import cv2
import numpy as np
import pandas as pd
import PyPDF2
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

from log_module import create_logger


class Globals:
    config_path = 'config.json'
    config = json.load(open(config_path))
    classifier_model_path = config['classifier_model_path']
    classifier_model = pickle.load(open(classifier_model_path, 'rb'))


g = Globals()


class PDFToHTML:
    def __init__(self, file_path):
        self.file_path = file_path
        print('\n########## Starting parsing for %s ##########\n' % file_path)
        self.filename = os.path.basename(file_path).split('.')[0]
        self.output_path = g.config['output_folder']
        self.file_folder = '%s/%s' % (self.output_path, self.filename)

        self.text_folder = '%s/%s' % (self.file_folder, 'texts')
        self.images_folder = '%s/%s' % (self.file_folder, 'images')
        self.html_folder = '%s/%s' % (self.file_folder, 'html-extra')
        self.tables_folder = '%s/%s' % (self.file_folder, 'tables-and-html')
        self.temp_folder = '%s/%s' % (self.file_folder, 'temp')
        self.delete_temp = g.config['delete_temp']

        self.model_prob_cutoff = g.config['model_prob_cutoff']  # NLP model prob cutoff
        self.choose_top_k = g.config['choose_top_k']  # Top items in dataframe
        self.imp_tb_types = g.config['imp_tb_types']  # Others are 'note' 'plaintext'

        with open(self.file_path, 'rb') as f:
            self.pages = PyPDF2.PdfFileReader(f).numPages
        self.test_page_num = min(g.config['test_page_num'], self.pages)
        self.pdf_type = None
        self.table_pages = []  # List containing tuple of (p_num, tb_type)

        # Multiprocessing settings
        self.pool_size = g.config['pool_size']
        self.generate_images_dpi = g.config['generate_images_dpi']
        self.MIN_COL_WIDTH = g.config['MIN_COL_WIDTH']
        self.MIN_ROW_WIDTH = g.config['MIN_ROW_WIDTH']

    def create_all_dirs(self, delete_old=True):
        PDFToHTML.create_folder(self.output_path, delete_old=False)
        PDFToHTML.create_folder(self.file_folder, delete_old=delete_old)
        PDFToHTML.create_folder(self.text_folder, delete_old=delete_old)
        # PDFToHTML.create_folder(self.pdf_folder, delete_old=delete_old)
        PDFToHTML.create_folder(self.images_folder, delete_old=delete_old)
        PDFToHTML.create_folder(self.html_folder, delete_old=delete_old)
        PDFToHTML.create_folder(self.tables_folder, delete_old=delete_old)
        PDFToHTML.create_folder(self.temp_folder, delete_old=delete_old)

    @staticmethod
    def delete_contents(path, delete_files=True, delete_folders=True, delete_itself=False):
        if os.path.isdir(path):
            if delete_itself:
                shutil.rmtree(path)
            else:
                if delete_files:
                    files = glob.glob('%s/*.*' % path)
                    for f in files:
                        os.remove(f)
                if delete_folders:
                    folders = glob.glob('%s/*' % path)
                    for f in folders:
                        os.rmdir(f)

    @staticmethod
    def create_folder(directory, delete_old=False):
        if delete_old:
            PDFToHTML.delete_contents(directory, delete_itself=True)
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print('Directory created. ' + directory)
        except OSError:
            print('Directory exists. ' + directory)

    @staticmethod
    def preprocessor(x):
        x = re.sub(r'[0-9\,]+', 'num', x)
        return re.sub(r'[\n()]+', ' ', x).lower()

    @staticmethod
    def get_command_output(cmd):
        return os.popen(cmd).read()

    def save_table_pages(self):
        df = pd.DataFrame(self.table_pages, columns=['page', 'prob', 'type', 'consolidated', 'is_note'])
        df = df[df.prob > self.model_prob_cutoff]
        df['type'] = df['type'].apply(lambda x: x.replace('is', 'IncomeStatement').replace('bs', 'BalanceStatement').replace('cf', 'CashflowStatement'))
        # TODO: Add more logic for filtering consolidated tables and schedules and notes
        df = df.sort_values(['type', 'prob', 'consolidated'], ascending=False).groupby('type').head(self.choose_top_k)
        df = df.groupby('type').head().sort_values(['consolidated'], ascending=False)
        df.to_csv(self.table_pages_path, index=False)
        print(df)

    def get_font_count_in_pdf(self):
        first_page = 1
        last_page = min(5, self.pages)
        cmd_out = PDFToHTML.get_command_output('pdffonts -f %s -l %s %s' % (first_page, last_page, self.file_path))
        text_file_path = '%s/%s-fonts.txt' % (self.output_path, self.filename)
        print('##### Fonts #####\n%s' % cmd_out)
        with open(text_file_path, "w", encoding='UTF-8') as f:
            f.write(cmd_out)

        with open(text_file_path, "r", encoding='UTF-8') as f:
            font_count = len(f.readlines()) - 2
            print('No. of fonts: %s' % font_count)
            return font_count

    def make_logger(self):
        self.logger = create_logger(level='DEBUG', log_folder='%s/logs' % self.file_folder, file_name=self.filename, do_print=True)

    def set_pdf_type3(self):
        xml_path = '%s/%s.xml' % (self.temp_folder, self.filename)
        os.system("pdftohtml -c -hidden -xml -f %s -l %s %s %s" % (self.test_page_num, self.test_page_num, self.file_path, xml_path))
        xmltree, xmlroot = read_xml(xml_path)
        # parse xml and generate a dict of pages
        pages = parse_pages(xmlroot)

        if len(pages[self.test_page_num]['texts']) > 0:
            image_path = '%s/%s-%s_1.png' % (self.temp_folder, self.filename, self.test_page_num)
            v = os.system('tesseract --oem 1 -l eng --psm 6 %s %s pdf' % (image_path, image_path))
            if v == 0:
                self.pdf_type = 'sandwich'
            else:
                self.pdf_type = 'normal'
        else:
            self.pdf_type = 'image'

        print('PDF type:', self.pdf_type)

    def set_pdf_type1(self):
        if self.get_font_count_in_pdf() < 2:
            self.pdf_type = 'image'
        else:
            self.pdf_type = 'text'

    def set_pdf_type2(self):
        if len(self.get_page_text2(self.file_path)) < 10:
            self.pdf_type = 'image'
        else:
            self.pdf_type = 'text'

    def get_page_texts1(self, file_path, p_num=5):
        text_file_path = '%s/%s-%s.txt' % (self.text_folder, self.filename, p_num)
        os.system("pdftotext -f %s -l %s %s %s" % (p_num, p_num, file_path, text_file_path))
        # os.system("pdftotext -f %s -l %s %s %s -layout" % (p_num, p_num, file_path, text_file_path))
        with open(text_file_path, 'r', encoding='UTF-8') as f:
            return f.read()

    def get_page_text2(self, file_path, p_num=5):
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            return pdf_reader.getPage(p_num).extractText()

    def generate_images(self):
        print('Generating images')
        convert_from_path(self.file_path, dpi=150, output_folder=self.images_folder, first_page=1, last_page=self.pages, fmt='png', thread_count=8)
        # os.system("pdftohtml -c -hidden -xml -fmt png %s %s/%s.xml" % (self.file_path, self.images_folder, self.filename))
        print('Generating images completed')

    def multithread_processor(self, to_pdf=False, to_text=False, gen_images=False):
        def image_to_pdf(image_paths):
            for image_path in image_paths:
                print(image_path)
                if self.pdf_type == 'image':
                    filename = '%s/%s-%s_1' % (self.images_folder, self.filename, image_path.split('.')[-2].split('-')[-1])
                else:
                    filename = image_path.split('.png')[0]
                print(filename)
                os.system('tesseract --oem 1 -l eng --psm 6 %s %s pdf' % (image_path, filename))
            return 0

        def image_to_text(image_paths):
            for image_path in image_paths:
                print(image_path)
                filename = '%s/%s' % (self.images_folder, image_path.split('.')[0])
                print(filename)
                os.system('tesseract --oem 1 -l eng --psm 6 %s %s' % (image_path, filename))
            return 0

        def generate_images(pages_list):
            for p_num in pages_list:
                print('Generating images %s' % p_num)
                convert_from_path(self.file_path, dpi=self.generate_images_dpi, output_folder=self.images_folder, first_page=p_num, last_page=p_num, fmt='png')
                print('Generating images completed %s' % p_num)
            return 0

        if to_pdf:
            paths = glob.glob('%s/*.png' % self.images_folder)
            print(paths)

            def multi_run_wrapper(args):
                return image_to_pdf(*args)

        elif to_text:
            paths = glob.glob('%s/*.png' % self.images_folder)

            def multi_run_wrapper(args):
                return image_to_text(*args)

        elif gen_images:
            paths = list(range(1, self.pages + 1))  # pages_list

            def multi_run_wrapper(args):
                return generate_images(*args)

        def divide_range(seq, num):
            avg = len(seq) / float(num)
            out = list()
            last = 0.0
            while last < len(seq):
                out.append([int(last), int(last + avg)])
                last += avg
            return out

        arg_data = list()

        for n in divide_range(range(len(paths)), self.pool_size):
            final_list = paths[n[0]:n[1]]
            arg_data.append([final_list])

        pool = Pool(self.pool_size)
        response_data = pool.map(multi_run_wrapper, arg_data)
        pool.close()
        pool.join()
        print('Done multiprocessing')

    def extract_table_type_list(self):
        self.table_pages_path = '%s/table_pages_pdftype_%s.csv' % (self.file_folder, self.pdf_type)
        if self.pdf_type in ['normal', 'sandwich']:
            with open(self.file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfFileReader(f)
                for p_num in range(1, self.pages + 1):
                    try:
                        text = self.get_page_texts1(self.file_path, p_num)
                        text = self.preprocessor(text)
                        p = np.round(np.max(g.classifier_model.predict_proba([text])), 3)
                        tb_type = g.classifier_model.predict([text])[0]
                    except Exception as e:
                        # Failure case when pdf is absent or not readable
                        p = 0
                        tb_type = 'Failed'
                        self.logger.error('Failed extracting text for page no: %s | %s' % (p_num, e))

                    if tb_type in self.imp_tb_types:
                        is_consolidated = 'consolidated' in text
                        is_notes = 'notes to' in text or 'schedules' in text
                        print('--> Found %s on page: %s | Prob: %s | consolidated: %s | notes: %s' % (tb_type, p_num, p, is_consolidated, is_notes))
                        self.table_pages.append((p_num, p, tb_type, is_consolidated, is_notes))
                    else:
                        print(p_num, p, tb_type)

            self.save_table_pages()

        else:
            # self.generate_images()
            self.multithread_processor(gen_images=True)
            self.multithread_processor(to_pdf=True)
            # for p_num in range(1, self.pages + 1):
            #     file_path = '%s/%s-%s_1.pdf' % (self.images_folder, self.filename, p_num)
            for file_path in glob.glob('%s/*.pdf' % self.images_folder):
                try:
                    text = self.get_page_texts1(file_path, p_num=1)
                    text = self.preprocessor(text)
                    p = np.max(g.classifier_model.predict_proba([text]))
                    tb_type = g.classifier_model.predict([text])[0]
                except Exception as e:
                    # Failure case when pdf is absent or not readable
                    p = 0
                    tb_type = 'Failed'
                    self.logger.error('Failed extracting text for: %s | %s' % (file_path, e))

                if tb_type in self.imp_tb_types:
                    is_consolidated = 'consolidated' in text
                    is_notes = 'notes to' in text or 'schedules' in text
                    print('--> Found %s on page: %s | Prob: %s | consolidated: %s | notes: %s' % (tb_type, file_path, p, is_consolidated, is_notes))
                    self.table_pages.append((file_path, p, tb_type, is_consolidated, is_notes))
                else:
                    print(file_path, p, tb_type)

            self.save_table_pages()


    def get_sandwich_pdf_paths(self):
        self.table_pages_path = '%s/table_pages_pdftype_%s.csv' % (self.file_folder, self.pdf_type)
        sandwich_pdf_paths = []
        df = pd.read_csv(self.table_pages_path)
        if self.pdf_type == 'normal':
            sandwich_pdf_paths = [(self.file_path, page) for page in df['page']]
        if self.pdf_type == 'image':
            for p_num, tb_type in df[['page', 'type']].values:
                if tb_type in self.imp_tb_types:
                    sandwich_pdf_paths.append((p_num, 1))
        elif self.pdf_type == 'sandwich':
            for p_num, tb_type in df[['page', 'type']].values:
                if tb_type in self.imp_tb_types:
                    sandwich_pdf_paths.append((self.file_path, p_num))

        print(sandwich_pdf_paths)

        return sandwich_pdf_paths

    def do_tablextract(self, pdf_path, p_num):
        if self.pdf_type == 'normal':
            print(pdf_path, p_num)
            tables = read_pdf(pdf_path, pages=[p_num], multiple_tables=True, java_options='-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider')
            for i in range(len(tables)):
                tables[i].fillna('').to_csv('%s/%s-%s.csv' % (self.tables_folder, p_num, i), encoding='utf-8')
                tables[i].fillna('').to_html('%s/%s-%s.html' % (self.tables_folder, p_num, i))
        else:
            filename = os.path.basename(pdf_path).split('.')[0].split('/')[0]
            DATAPATH = self.images_folder  # 'data/'
            INPUT_XML = '%s/%s.xml' % (self.images_folder, filename)
            os.system("pdftohtml -c -hidden -xml -enc UTF-8  -f %s -l %s %s %s" % (p_num, p_num, pdf_path, INPUT_XML))
            os.system("pdftohtml -c -hidden -f %s -l %s %s %s/%s.html" % (p_num, p_num, pdf_path, self.html_folder, filename))

            # Load the XML that was generated with pdftohtml
            xmltree, xmlroot = read_xml(INPUT_XML)
            # parse it and generate a dict of pages
            pages = parse_pages(xmlroot)
            # print(pages[p_num]['texts'][0])
            p = pages[p_num]

            # Detecting lines
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
            MIN_COL_WIDTH = self.MIN_COL_WIDTH  # minimum width of a column in pixels, measured in the scanned pages
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
            MIN_ROW_WIDTH = self.MIN_ROW_WIDTH  # minimum width of a row in pixels, measured in the scanned pages
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

            html_output_file = os.path.join(self.tables_folder, filename + '.html')
            print("saving extracted data to '%s'" % html_output_file)
            df.to_html(html_output_file, index=False, header=False)

    def extract_and_save_tables(self):
        sandwich_pdf_paths = self.get_sandwich_pdf_paths()

        for pdf_path, p_num in sandwich_pdf_paths:
            try:
                self.do_tablextract(pdf_path, p_num)
            except Exception as e:
                self.logger.error('\n########\nFailed do_tablextract for: %s | %s\n########\n' % (pdf_path, e))

    def cleanup(self):
        if self.delete_temp:
            self.delete_contents(self.temp_folder, delete_files=True, delete_folders=True, delete_itself=True)


if __name__ == '__main__':
    input_folder = g.config['input_folder']
    for file_path in glob.glob('%s/*.pdf' % input_folder):
        parser = PDFToHTML(file_path)
        parser.create_all_dirs(delete_old=True)
        parser.make_logger()
        parser.set_pdf_type3()
        parser.extract_table_type_list()
        parser.extract_and_save_tables()
        parser.cleanup()
        print('xxxxxxxxxxx \nCOMPLETED %s \nxxxxxxxxxxx' % file_path)
