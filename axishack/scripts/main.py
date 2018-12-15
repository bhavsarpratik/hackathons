# -*- coding: utf-8 -*-

import glob
import os
import pickle
import re
import shutil
from math import ceil, degrees, floor, radians

import cv2
import numpy as np
import pandas as pd
import PyPDF2
from pdf2image import convert_from_path
from pdftabextract.common import parse_pages, read_xml

import utilities as util
from multi_processor import multithread_processor
from tablextract import do_tablextract

logger = util.create_logger(level='DEBUG', log_folder='logs', file_name='logs', do_print=True)


class Globals:
    pass


g = util.add_params_to_object_from_dict_path(Globals(), 'config.json')


class PDFToExcel:
    def __init__(self, file_path):
        self.file_path = file_path
        print('\n########## Starting parsing for %s ##########\n' % file_path)
        self.filename = os.path.basename(file_path).split('.')[0]
        self.extension = self.file_path.split('.')[-1]
        self.output_path = g.output_folder
        self.file_folder = '%s/%s' % (self.output_path, self.filename)

        self.text_folder = '%s/%s' % (self.file_folder, 'texts')
        self.images_folder = '%s/%s' % (self.file_folder, 'images')
        # self.html_folder = '%s/%s' % (self.file_folder, 'html-extra')
        self.tables_folder = '%s/%s' % (self.file_folder, 'tables')
        self.tables_folder_tabula = '%s/%s/%s' % (self.file_folder, 'tables', 'tabula')
        self.tables_folder_camelot = '%s/%s/%s' % (self.file_folder, 'tables', 'camelot')
        self.temp_folder = '%s/%s' % (self.file_folder, 'temp')

        if self.extension == 'pdf':
            self.doc_type = 'pdf'
            with open(self.file_path, 'rb') as f:
                self.pages = PyPDF2.PdfFileReader(f).numPages
        elif self.extension in ['doc', 'docx']:
            self.doc_type = 'msword'
            self.pages = 1
        else:
            self.doc_type = 'image'
            self.pages = 1
        print('No. of pages %s' % self.pages)
        self.test_page_num = ceil(0.7 * self.pages)
        self.pdf_type = 'NA'
        self.table_pages = []  # List containing tuple of (p_num, tb_type)

    def create_all_dirs(self, delete_old=True):
        util.create_folder(self.output_path, delete_old=False)
        util.create_folder(self.file_folder, delete_old=delete_old)
        util.create_folder(self.text_folder, delete_old=delete_old)
        util.create_folder(self.images_folder, delete_old=delete_old)
        # util.create_folder(self.html_folder, delete_old=delete_old)
        util.create_folder(self.tables_folder, delete_old=delete_old)
        util.create_folder(self.tables_folder_tabula, delete_old=delete_old)
        util.create_folder(self.tables_folder_camelot, delete_old=delete_old)
        util.create_folder(self.temp_folder, delete_old=delete_old)

    def make_logger(self):
        self.logger = util.create_logger(level='DEBUG', log_folder='%s/logs' % self.file_folder, file_name=self.filename, do_print=True)

    def set_pdf_type(self):
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

        self.logger.info('PDF type:%s checked on page %s' % (self.pdf_type, self.test_page_num))

    def get_page_text(self, file_path, p_num=5):
        text_file_path = '%s/%s-%s.txt' % (self.text_folder, self.filename, p_num)
        os.system("pdftotext -f %s -l %s %s %s" % (p_num, p_num, file_path, text_file_path))
        # os.system("pdftotext -f %s -l %s %s %s -layout" % (p_num, p_num, file_path, text_file_path))
        with open(text_file_path, 'r', encoding='UTF-8') as f:
            return f.read()

    def save_table_pages(self):
        df = pd.DataFrame(self.table_pages, columns=['page'])
        df.to_csv(self.table_pages_path, index=False)
        print(df)

    def extract_table_type_list(self):
        self.table_pages_path = '%s/table_pages_pdftype_%s.csv' % (self.file_folder, self.pdf_type)
        if self.doc_type == 'pdf':
            if self.pdf_type in ['normal', 'sandwich']:
                for p_num in range(1, self.pages + 1):
                    self.table_pages.append((p_num))

                self.save_table_pages()

            elif self.pdf_type == 'image':
                multithread_processor(self, g, gen_images=True)
                multithread_processor(self, g, to_pdf=True)
                for file_path in glob.glob('%s/*.pdf' % self.images_folder):
                    self.table_pages.append((file_path))

                self.save_table_pages()

        if self.doc_type == 'image':
            image_path = self.file_path
            print(image_path)
            # filename = image_path.split('.')[0]
            filename = '%s/%s' % (self.images_folder, self.filename)
            print(filename)
            os.system('tesseract --oem 1 -l eng --psm 6 %s %s pdf' % (image_path, filename))

            for file_path in glob.glob('%s/*.pdf' % self.images_folder):
                self.table_pages.append((file_path))

            self.save_table_pages()

    def get_sandwich_pdf_paths(self):
        self.table_pages_path = '%s/table_pages_pdftype_%s.csv' % (self.file_folder, self.pdf_type)
        sandwich_pdf_paths = []
        df = pd.read_csv(self.table_pages_path)
        if self.pdf_type == 'normal':
            sandwich_pdf_paths = [(self.file_path, page) for page in df['page']]
        elif self.pdf_type == 'image' or self.doc_type == 'image':
            for path in df['page'].values:
                sandwich_pdf_paths.append((path, 1))
        elif self.pdf_type == 'sandwich':
            for p_num in df['page'].values:
                sandwich_pdf_paths.append((self.file_path, p_num))

        print(sandwich_pdf_paths)
        return sandwich_pdf_paths

    def extract_and_save_tables(self):
        sandwich_pdf_paths = self.get_sandwich_pdf_paths()

        for pdf_path, p_num in sandwich_pdf_paths:
            try:
                do_tablextract(self, g, pdf_path, p_num)
            except Exception as e:
                self.logger.error('\n########\nFailed do_tablextract for: %s | %s\n########\n' % (pdf_path, e))


    def combine_and_save_all_files(self):
        print('Generating Tabula combined output')
        path = self.tables_folder_tabula
        file_paths = glob.glob('%s/*.csv' % path)
        if file_paths:
            tabula_combine_output_path = os.path.join(self.output_path, self.filename+'-tabula.csv')
            with open(tabula_combine_output_path, 'w', encoding='utf-8') as out_file:
                for file_path in file_paths:
                    with open(file_path, 'r', encoding='utf-8') as in_file:
                        out_file.write(file_path + '\n')
                        for line in in_file.readlines():
                            out_file.write(line)
                        out_file.write('\n')
                        out_file.write('\n')

        print('Generating camelot combined output')
        path = self.tables_folder_camelot
        file_paths = glob.glob('%s/*.csv' % path)
        if file_paths:
            camelot_combine_output_path = os.path.join(self.output_path, self.filename+'-camelot.csv')
            with open(camelot_combine_output_path, 'w', encoding='utf-8') as out_file:
                for file_path in file_paths:
                    with open(file_path, 'r', encoding='utf-8') as in_file:
                        out_file.write(file_path + '\n')
                        for line in in_file.readlines():
                            out_file.write(line)
                        out_file.write('\n')
                        out_file.write('\n')

        print('Generating tablextract combined output')
        path = self.tables_folder
        file_paths = glob.glob('%s/*.csv' % path)
        if file_paths:
            combine_output_path = os.path.join(self.output_path, self.filename+'-tablextract.csv')
            with open(combine_output_path, 'w', encoding='utf-8') as out_file:
                for file_path in file_paths:
                    with open(file_path, 'r', encoding='utf-8') as in_file:
                        out_file.write(file_path + '\n')
                        for line in in_file.readlines():
                            out_file.write(line)
                        out_file.write('\n')
                        out_file.write('\n')

    def cleanup(self):
        if g.delete_temp:
            util.delete_contents(self.temp_folder, delete_files=True, delete_folders=True, delete_itself=True)
            util.delete_contents(self.images_folder, delete_files=True, delete_folders=True, delete_itself=True)


if __name__ == '__main__':
    input_folder = g.input_folder
    for file_path in glob.glob('%s/*.*' % input_folder):
        try:
            parser = PDFToExcel(file_path)
            parser.create_all_dirs(delete_old=True)
            parser.make_logger()
            if parser.doc_type == 'pdf':
                parser.set_pdf_type()
            parser.extract_table_type_list()
            parser.extract_and_save_tables()
            parser.combine_and_save_all_files()
            # parser.cleanup()
            logger.info('xxxxxxxxxxx \nCOMPLETED %s \nxxxxxxxxxxx' % file_path)
        except Exception as e:
            logger.error('%s: %s' % (file_path, e))
