import glob
import os
from multiprocessing.dummy import Pool

from pdf2image import convert_from_path


def multithread_processor(self, g, to_pdf=False, to_text=False, gen_images=False):
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
            convert_from_path(self.file_path, dpi=g.generate_images_dpi, output_folder=self.images_folder, first_page=p_num, last_page=p_num, fmt='png')
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

    for n in divide_range(range(len(paths)), g.pool_size):
        final_list = paths[n[0]:n[1]]
        arg_data.append([final_list])

    pool = Pool(g.pool_size)
    response_data = pool.map(multi_run_wrapper, arg_data)
    pool.close()
    pool.join()
    print('Done multiprocessing')
