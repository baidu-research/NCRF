import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from wsi.data.annotation import Formatter  # noqa

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('xml_path', default=None, metavar='XML_PATH', type=str,
                    help='Path to the directory of Camelyon16 xml files')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the directory of json files')


def run(args):
    xml_files = [f for f in os.listdir(args.xml_path)]

    if not os.path.exists(args.json_path):
        os.mkdir(args.json_path)

    for xml_file_name in xml_files:
        pid = xml_file_name.split('.')[0]
        inxml = os.path.join(args.xml_path, xml_file_name)
        outjson = os.path.join(args.json_path, pid + '.json')
        Formatter.camelyon16xml2json(inxml, outjson)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
