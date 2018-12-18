import datetime
import logging
import os


def create_folder(directory):
    import os
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Directory created. ' + directory)
    except OSError:
        print('Directory exists. ' + directory)


def create_logger(level='DEBUG', log_folder='logs', file_name=None, do_print=False):
    """Creates a logger of given level and saves logs to a file of __main__'s name

    LEVELS available
    DEBUG: Detailed information, typically of interest only when diagnosing problems.
    INFO: Confirmation that things are working as expected.
    WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. 'disk space low'). The software is still working as expected.
    ERROR: Due to a more serious problem, the software has not been able to perform some function.
    CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
    """
    # import __main__
    # if file_name is None:
    #     file_name = __main__.__file__.split('.')[0]

    logger = logging.getLogger(file_name)
    logger.setLevel(getattr(logging, level))
    formatter = logging.Formatter(
        '%(asctime)s:%(levelname)s:%(module)s:%(funcName)s: %(message)s', "%Y-%m-%d %H:%M:%S")
    stream_formatter = logging.Formatter(
        '%(levelname)s:%(module)s:%(funcName)s: %(message)s')

    # formatter = logging.Formatter('%(message)s')
    # stream_formatter = logging.Formatter('%(message)s')

    date = datetime.date.today()
    date = '%s-%s-%s' % (date.day, date.month, date.year)
    log_file_path = os.path.join(log_folder, '%s-%s.log' % (file_name, date))

    create_folder(log_folder)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    if do_print:
        logger.addHandler(stream_handler)

    return logger
