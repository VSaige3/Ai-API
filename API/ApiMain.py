from os import PathLike
from typing import Union, List, Tuple

import csv
import json
import sys
import os.path
import numpy as np
import io
from xml.etree import ElementTree
from io import StringIO
import API.Network
import atexit
from keras.models import Model, load_model
# fields we will end up with:
# format, rows included, cols included, identification column and row
# turn that into a 2D array with the minimal list of values and names
import API
from API.ProgramXml import XMLReader

stat_defaults = {"xml_file": "../XML/API-metadata.xml",
                 "metadata_format": "../XML/metadata-format.xml"}


class DataProcessor:
    _required_keys = ['format', 'data']
    _possible_formats = ['xml', 'xlsx', 'csv']
    _default_data_cvs: dict = {
        'row_id_col': 0,
        'col_id_row': 0,
        'rename_cols': {},
        'rename_rows': {},
        'swap_axes': False
    }

    _required_data_cvs: list = [
        'data_range',
        ('cols_used', 'cols_unused'),
        ('rows_used', 'rows_unused'),
    ]
    _data_types_cvs: dict = {
        'data_range': list,
        'cols_used': (list, int),
        'cols_unused': (list, int),
        'rows_used': (list, int),
        'rows_unused': (list, int),
        'row_id_col': int,
        'col_id_row': int,
        'rename_cols': dict,
        'rename_rows': dict,
        'swap_axes': bool
    }

    defaults: dict = {}

    settings: dict = {}

    # _settings: dict = {}
    #
    # @property
    # def settings(self) -> dict:
    #     """
    #     gets the settings of the DataProcessor that are static throught classes
    #     Might add some way of creating multiple, but that is not needed yet
    #
    #     :return: a settings object loaded from the file
    #     """
    #     temp = self.load_processor("../API-metadata.xml", "../metadata-format.xml")
    #     s = temp._settings
    #     del temp
    #     # return s
    #     return self._settings
    #
    # @settings.setter
    # def settings(self, value):
    #     self._settings = value
    #     #TODO: make it so that it writes to the file

    _datasets = []

    @property
    def datasets(self) -> List[Tuple]:
        """
        represents the datasets in this Collector object
        Setting it automatically changes the files

        :return: the variable _datasets
        """
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        self._datasets = value
        # TODO: update file automatically as well as settings

    _dtypes = ["econ", "demo", "geo", "viral", "y"]

    def __init__(self, xml_file: Union[str, None] = None,
                 metadata_format: Union[str, None] = None,
                 load=True):
        if not load:
            return

        if not xml_file:
            xml_file = stat_defaults["xml_file"]
        if not metadata_format:
            metadata_format = stat_defaults["metadata_format"]

        self.xml_file = xml_file
        self.metadata_format = metadata_format
        self.load(xml_file, metadata_format)

    def save(self):
        DataProcessor.save_processor(self, self.xml_file, self.metadata_format)

    def load(self, filepath, md_format):
        dp = DataProcessor.load_processor(filepath, md_format)
        self.settings = dp.settings
        self.datasets = dp.datasets
        self.defaults = dp.defaults

    @staticmethod
    def load_processor(filepath, md_format):
        md = ElementTree.parse(filepath).getroot()
        fmat = ElementTree.parse(md_format).getroot()

        # def match(root: Union[ElementTree.ElementTree, ElementTree.Element],
        #           expr,
        #           tag: Union[ElementTree.ElementTree, ElementTree.Element, None] = None, attr='tag'):
        #     if tag is None:
        #       tag = root
        #     m = root.find(expr)
        #
        #     if getattr(tag, attr) == expr:
        #         return tag
        #
        #     if m:
        #         return m
        #     else:
        #         if len(list(tag)) > 0:
        #             for x in tag:
        #                 if getattr(x, attr) == getattr(tag, attr):
        #                     continue
        #                 else:
        #                     print(x)
        #                     m = match(root, expr, x)
        #                     if m:
        #                         return m
        #         else:
        #             print("no sub")
        #             return None

        vals = XMLReader.get_vals(md, fmat,
                                  ["(METADATA_DPATH)", "(FILEDATA_DPATH)", "(DBMETA_PATH)", "settings", "databases"])
        dp = DataProcessor(load=False)
        dp.defaults.update({"metadata_path": vals[0]})
        dp.defaults.update({"filedata_path": vals[1]})
        dp.defaults.update({"db_metadata_path": vals[2]})

        for x in vals[3].findall("database"):
            dp.add_database(x.attrib["path"])

        for x in vals[3].findall("dataset"):
            dp.add_dataset(x.attrib["path"])

        print(vals)
        _settings = vals[4].findall("setting")
        print(_settings)
        for s in _settings:
            dp.settings.update({s.attrib['name']: s.attrib['value']})

        return dp

    @staticmethod
    def save_processor(data_collecter, filepath, format):
        assert isinstance(data_collecter, DataProcessor)
        xml_proto = ElementTree.TreeBuilder()
        # TODO: build tree from format

    @staticmethod
    def _database_formatted(database_path, dbmetadata=None) -> (bool, BaseException):
        """
        check if database is formatted correctly

        :param database_path: the path to the database folder
        :param dbmetadata: the filename of the database metadata file
        :return: a boolean representing whether it is formatted, and an exception representing why it is not.
        """

        if not dbmetadata:
            dbmetadata = XMLReader.get_vals(stat_defaults["xml_file"],
                                            stat_defaults["metadata_format"],
                                            ["(DBMETA_PATH)"])[0]

        if not os.path.isdir(database_path):
            return False, NotADirectoryError("database path {} is not a directory".format(database_path))
        if not os.path.isfile(database_path + dbmetadata):
            return False, FileNotFoundError("could not find file {}".format(database_path + dbmetadata))

        return True, None

    def add_database(self, path, dbmetadata=None):
        """
        creates an entry into the stored databases provided it has the necessary files

        :raises FileNotFound: if one of the files needed for loading a database is not there
        :raises ValueError: if one of the parameters is in the wrong format

        :raises NotADirectory: if the given path is not a directory
        :param path: The path to the root of the database
        :param dbmetadata: Database metadata path; is set to none because it defaults to self.defaults["db_metadata_path"]
        :return: itself for building

        """
        # assign dbmetadata to defaults
        if not dbmetadata:
            dbmetadata = self.defaults["db_metadata_path"]

        formatted, exc = self._database_formatted(path)

        # raise exception if not formatted correctly
        if not formatted:
            raise exc

        # if path does not exist
        if not os.path.exists(path):
            raise FileNotFoundError("database directory {} could not be located.\n"
                                    " it is possible that the name is missing a '/'")

        # we can't have path not being a folder
        if not os.path.isdir(path):
            raise NotADirectoryError("{} is not a folder".format(path))

        # add datasets
        dbdata = json.load(open("{}{}".format(path, dbmetadata)))
        for name, dset in dbdata.items():
            dpath = path + dset["path"]
            if os.path.exists(dpath):
                if os.path.isdir(dpath):
                    # just add each part as a pair
                    self.add_dataset(dpath, dset["variable"], dset["type"])

    def add_dataset(self, path, _variable=None, _type=None, _json=None):
        # actually add a dataset to the datasets
        if not _variable or not _type:
            if _json:
                assert hasattr(_json, "__getitem__")
                _variable = _json["variable"]
                _type = _json["type"]
            else:
                raise ValueError("not enough parameters supplied")

        if os.path.exists(path):
            self.datasets.append((path, _variable, _type))

    def __add__(self, other):
        assert isinstance(other, (str, bytes, PathLike))
        other = str(other)
        if not other.endswith("/"):
            otherd = other + "/"
            otherf = other
        else:
            otherd = other
            otherf = other.rstrip("/")
        if os.path.exists(otherd):
            if os.path.exists(otherd + self.defaults["db_metadata_path"]):
                self.add_database(otherd)
            elif os.path.exists(otherd + self.defaults["metadata_path"]):
                self.add_dataset(otherd)
        if os.path.exists(otherf):
            osp = otherf.split("/")
            if osp[-1] in list(self.defaults.values())[:2]:
                self.add_dataset(osp[:-1])
                print(osp)
                print(osp[:-1])
                print(osp[-1])
            elif osp[-1] in self.defaults[self.defaults[-1]]:
                self.add_database(osp[:-1])

    def collect_all(self):
        pass  # TODO: collect all the data and get the settings

    @staticmethod
    def is2DList(matrix_list) -> bool:
        if isinstance(matrix_list[0], list):
            return True
        else:
            return False

    @staticmethod
    def get_data(data: Union[str, List[List], io.IOBase, io.StringIO, np.ndarray],
                 metadata: Union[str, dict, io.FileIO], data_filling_mode="None") -> (List[list], list, list):
        """
        Gets the necessary data from the given data file or filepath and a metadata file


        :param data: A filepath, file, or 2d array representing the data.
        :param metadata: The metadata in a specific format (only supports json currently) Format can be found at (TODO: insert URL)
        :param data_filling_mode: one of "default", "none", or "average"
        :return: The data compressed into just the necessary rows and columns and 2 lists of row/column names in the format (data, rows, columns)
        """
        raw_data = data
        if isinstance(data, (str, bytes, bytearray)):
            # check if filepath
            if os.path.isfile(data):
                if os.path.exists(data):
                    raw_data = open(data, 'r')
                else:
                    raise FileNotFoundError("file {} not found".format(data))
            elif os.path.isdir(data):
                raise IsADirectoryError("Expected file, got directory")
        elif isinstance(data, (list, np.ndarray)):
            if DataProcessor.is2DList(data):
                raw_data = [", ".join(row) for row in data]
        else:
            # assume is a file
            raw_data = data

        form, err, metadata = DataProcessor.is_formatted(metadata)
        # check for error
        if err:
            raise ValueError(err)
        # fill in defaults
        metadata = DataProcessor._fill_defaults_safe(metadata)
        meta_data = metadata['data']
        # deduce data format and send to get_datas? No, use different method because need a file not a string

        reader = csv.reader(raw_data)
        read_arr = [x for x in reader]

        # NOTE: row labels is a column labeling rows, and vice versa
        # Collect correct row

        row_lbls = [row[meta_data['row_id_col']] if len(row) - 1 >= meta_data['row_id_col'] else None for row in
                    read_arr]
        col_lbls = read_arr[meta_data['col_id_row']]
        print("row labels: {}\ncol labels: {}".format(row_lbls, col_lbls))
        # replace labels with metadata-defined ones
        if 'rename_rows' in meta_data:
            print("renaming rows...\n")
            for i, x in enumerate(row_lbls):
                if x in meta_data['rename_rows']:
                    row_lbls[i] = meta_data['rename_rows'].get(x)

        if 'rename_cols' in meta_data:
            print("renaming columns...\n")
            for i, x in enumerate(col_lbls):
                if x in meta_data['rename_cols']:
                    col_lbls[i] = meta_data['rename_cols'].get(x)

        print("columns labeled:\n {}\n rows labeled:\n {}".format(col_lbls, row_lbls))

        # only use rows in rows_used
        ret: list
        dat_range = meta_data['data_range']
        # confine to just data range
        print(np.array(read_arr).shape)
        ret = read_arr[dat_range[1]:dat_range[1] + dat_range[3]]
        print(np.array(ret).shape)
        ret = [x[dat_range[0]:dat_range[0] + dat_range[2]] for x in ret]
        print(np.array(ret).shape)
        print("new cols: {}\nnew rows: {}".format(col_lbls, row_lbls))
        # Get only useable data
        # TODO: make numbers relative to data range, is current prob.

        # select rows
        # ret = [x if (i + dat_range[0]) in meta_data['rows_used'] else None for i, x in enumerate(ret)]
        ret = [x if i - 2 in meta_data['rows_used'] else None for i, x in enumerate(ret)]
        [ret.remove(None) for _ in range(ret.count(None))]
        print(np.array(ret).shape)

        # select columns
        # ret = [[x if (i + dat_range[1]) in meta_data['cols_used'] else None for i, x in enumerate(row)] for row in ret]
        ret = [[x if i + 1 in meta_data['cols_used'] else None for i, x in enumerate(row)] for row in ret]
        [[row.remove(None) for _ in range(row.count(None))] for row in ret]
        print(np.array(ret).shape)

        # remove unused labels
        row_lbls = [x if i in meta_data['rows_used'] else None for i, x in enumerate(row_lbls)]
        [row_lbls.remove(None) for _ in range(row_lbls.count(None))]
        print(f"row labels: {row_lbls}")

        col_lbls = [x if i in meta_data['cols_used'] else None for i, x in enumerate(col_lbls)]
        [col_lbls.remove(None) for _ in range(col_lbls.count(None))]

        if data_filling_mode == "average" or data_filling_mode == "avg":
            ret = [DataProcessor.fill_NAN(x) for x in ret]

        return ret, row_lbls, col_lbls

    @staticmethod
    def get_datas(data: Union[str, bytearray, bytes, List[chr]], metadata: Union[str, dict, io.FileIO]) -> (
            List[list], list, list):
        """
        Gets data from a string of comma separated values or whatever format it is in (currently only supports CSV)

        :param metadata: the string or file that represents the metadata
        :param data: The string representing the data
        :return: a 2D array of the data with each part labeled
        """
        return DataProcessor.get_data(StringIO(str(data)), metadata)

    @staticmethod
    def get_data_from_dir(directory_path: Union[PathLike, str], data_file="{}.csv", fields="fields.json") -> (
            List[list], list, list):
        sep = "/"
        directory_path = str(directory_path)
        sep_path = directory_path.split(os.path.dirname(sep))
        print(sep_path)
        if "{}" in data_file:
            data_file = data_file.format(sep_path[len(sep_path) - 2])
            print(data_file)
        data_fullpath = directory_path + data_file
        fields_fullpath = directory_path + fields
        return DataProcessor.get_data(data_fullpath, fields_fullpath)

    @staticmethod
    def get_data_from_dir_file_data(directory_path: Union[PathLike, str], file_data="file_data.json") -> List[Tuple]:
        ret = []
        if os.path.exists(directory_path + file_data):
            file_datajson = json.load(open(directory_path + file_data))
            for ting in file_datajson:
                x = file_datajson[ting]
                print(f"opening file {x['path']}")
                if os.path.exists(x['path']):
                    if 'fields' in x:
                        if os.path.exists(x['fields']):
                            ret.append(DataProcessor.get_data(x['path'], x['fields']))
                        elif os.path.exists(directory_path + x['fields']):
                            ret.append(DataProcessor.get_data(x['path'], directory_path + x['fields']))
                        else:
                            raise FileNotFoundError("Could not find json fields file {}".format(x['fields']))
                    else:
                        ret.append(DataProcessor.get_data(x['path'], directory_path + 'fields.json'))
                        print(ret)
        else:
            raise FileNotFoundError("Could not find file: {}".format(directory_path + file_data))

        return ret

    @staticmethod
    def fill_NAN(row: Union[np.ndarray, list], none_delim=0, func=lambda row: np.average(row)) -> list:
        # If somebody inputs model.predict as their function, you can use machine learning to predict
        row = [x for x in row]
        [row.remove(none_delim) for _ in range(row.count(none_delim))]
        fill = func(row)
        row = [fill if x == none_delim else x for x in row]
        return row

    @staticmethod
    def is_formatted(metadata: Union[str, dict, io.FileIO]) -> (bool, Union[str, None], dict):
        """
        checks if metadata is formatted correctly.

        :param metadata: The metadata in almost any format, can be accepted as a file object, dictionary, filepath, or string
        :return: True if it is formated, false if it is not. also returns error message (if any) and metadata
        """
        # Make sure metadata is in the correct format
        metadata_dict: dict
        if isinstance(metadata, str):
            if os.path.isfile(metadata):
                if os.path.exists(metadata):
                    metadata_dict = json.load(open(metadata, 'r'))
                else:
                    raise FileNotFoundError("file {} not found".format(metadata))
            elif os.path.isdir(metadata):
                raise IsADirectoryError("Expected file, got directory")
            else:
                metadata_dict = json.loads(metadata)
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        elif isinstance(metadata, io.FileIO):
            metadata_dict = json.loads(str(metadata.readlines()))
        else:
            raise ValueError(
                "expected type <class 'dict'>, <class 'string'>, or file-like object, instead got {}".format(
                    type(metadata)))

        meta_keys = metadata_dict.keys()
        # Check for required fields
        if not all([x in meta_keys for x in DataProcessor._required_keys]):
            err = "missing required fields"
            return False, err, None
        if not metadata_dict['format'] in DataProcessor._possible_formats:
            err = ("not an acccepted 'format' tag : {}".format(metadata_dict['format']))
            return False, err, None

        data: dict = metadata_dict['data']
        # make sure the pairs are in the correct format
        if metadata_dict['format'] in ['xlsx', 'csv']:
            if not all([any([y1 in data for y1 in y]) if isinstance(y, tuple) else y in data for y in
                        DataProcessor._required_data_cvs]):
                err = ("data tag missing required fields,\n contains {} and should contain {}"
                       .format([x for x in data.keys()], DataProcessor._required_data_cvs))
                err += ([any([y1 in data for y1 in y]) if isinstance(y, tuple) else y in data for y in
                         DataProcessor._required_data_cvs])
                return False, err, None
            if not all([isinstance(data.get(x), DataProcessor._data_types_cvs.get(x)) if x in data else True for x in
                        DataProcessor._data_types_cvs]):
                err = ("datatypes do not match! {}".format(
                    [((data.get(x)), DataProcessor._data_types_cvs.get(x)) if x in data else "" for x in
                     DataProcessor._data_types_cvs]))
                return False, err, None
        return True, None, metadata_dict

    @staticmethod
    def _fill_defaults_safe(meta: dict) -> dict:
        """
        unlike fill_defaults, does not check that metadata is the correct type because i'm lazy

        WARNING: only supports cvs and xslx files (exel)

        :param meta: A dictionary of metadata json
        :return: A new dictionary with missing required values filled out
        """
        dat: dict = meta['data']
        dat_range: dict = dat['data_range']
        # fill in static defaults
        for def_key in DataProcessor._default_data_cvs:
            if def_key not in dat:
                dat[def_key] = DataProcessor._default_data_cvs[def_key]

        # fill in dynamic defaults
        if 'cols_unused' in dat:
            if 'cols_used' not in dat:
                dat['cols_used'] = [None if x in dat['cols_unused'] else x  # - dat['data_range'][0]
                                    for x in range(dat_range[0], dat_range[0] + dat_range[2])]
        if 'rows_unused' in dat:
            if 'rows_used' not in dat:
                dat['rows_used'] = [None if x in dat['rows_unused'] else x  # - dat['data_range'][1]
                                    for x in range(dat_range[1], dat_range[1] + dat_range[3])]
        elif 'rows_used' in dat:
            pass

        dat['data_range'] = dat_range
        meta['data'] = dat

        print(meta)

        return meta

    def close(self):
        self.save()


class CustomConsole:
    Collecter: Union[DataProcessor, None]
    Network: Union[Model, None]
    collected: bool = False
    running: bool = True
    delim = " "

    @property
    def _input_size(self):
        x = 12
        if self.Collecter:
            if "input_size" in self.Collecter.settings:
                print(self.Collecter.settings["input_size"])
                return self.Collecter.settings["input_size"]
            else:
                return [(x, x), (x, x), (x, x), (x, x)]
        else:
            return [(x, x), (x, x), (x, x), (x, x)]

    @_input_size.setter
    def _input_size(self, value):
        self._input_size = value
        if self.Collecter:
            self.Collecter.settings.update({"input_size": value})

    @_input_size.deleter
    def _input_size(self):
        del self._input_size

    def __init__(self):
        self.Network = None
        self.Collecter = None

    def run(self, argv):
        help_string = ""
#          help_string = "ApiMain [begin|end|options|help|add|create|exit]\n" \
#                       "ApiMain begin/-b collector/-C (-d/--directory <directory>|-f <file> <json>|--default/-df)" \
#                       " --output/-o <data output directory>\n" \
#                       "ApiMain begin/-b network/-N [train/-t|load/-l]\n" \
#                       "ApiMain end/-e\n" \
#                       "ApiMain options/-o \n" \
#                       "\t--verbose/-v\t0 - no feedback\n\t1 - errors and warnings\n\t2 - everything\n" \
#                       "\t--defaults/-d \n" \
#                       "\t\twhen called with no arguments," \
#                       " displays current default settings and databases, in the format\n" \
#                       "\t\t(name) | (current value(s)) | (possible values)\n" \
#                       "ApiMain create [collector|network]" \
#                       "\t\t alternately, one can specify ApiMain options <attribute> <new_value> to set an attribute" \
#                       "ApiMain --help/-h return this tooltip\n" \
#                       "ApiMain --add/-a -m/--metadata <filename> or --directory/-d <directory> add to default directory\n" \
#                       "ApiMain --create/-c --API-xml/-p-m <file> [<option> <value> <values/type> ...]\n" \
#                       "\t--create --default-metadata/-d-m <directory>"
        options = {
            "begin": ["begin", "-b"],
            "end": ["end", "-e"],
            "help": ["--help", "-h"],
            "options": ["options", "settings", "-o"],
            "directory": ["--directory", "-d"],
            "file": ["--file", "-f"],
            "output": ["--out", "--output", "-o"],
            "collector": ["collector", "-C"],
            "network": ["network", "-N"],
            "load": ["load", "-l"],
            "train": ["train", "-t"],
            "create": ["create", "-C"],
            "exit": ["exit"]
        }
        if not argv:
            uargs = sys.argv[1:]
        else:
            uargs = argv

        ulen = len(uargs)

        arr = []
        if '|' in uargs:
            for x in uargs:
                if x != '|':
                    arr.append(x)
                else:
                    self.run(arr)
                    arr = []

            self.run(arr)
            return

        if uargs[0] in options["begin"]:
            if ulen > 1:

                if uargs[1] in options["collector"]:
                    # Add more functionality
                    print("starting collector")

                elif uargs[1] in options["network"]:
                    if ulen > 2:
                        if uargs[2] in options["train"]:
                            if self.Network:
                                if self.collected:
                                    print("running network")
                                else:
                                    print("cannot run network, there is no data collected")
                            else:
                                print("cannot, run network, is has not been loaded")

                        elif uargs[2] in options["load"]:
                            if ulen > 3:
                                pass  # load from given file
                            elif self.Collecter and "network_save_path" in self.Collecter.settings:
                                self.Network = load_model(self.Collecter.settings["network_save_path"])
                            elif os.path.exists("./Network_default"):
                                self.Network = load_model("./Network_default")
                            else:
                                print("no place to load network from")
                        else:
                            print("unrecognized command {}".format(uargs[2]))

        elif uargs[0] in options["create"]:
            if ulen > 1:
                if uargs[1] in options["network"]:
                    self.Network = API.Network.create_network(self._input_size)
                elif uargs[1] in options["collector"]:
                    self.Collecter = DataProcessor()

        elif uargs[0] in options["options"]:
            if self.Collecter:
                if ulen == 3:
                    self.Collecter.settings.update({uargs[1]: uargs[2]})
                elif ulen == 2:
                    print(self.Collecter.settings[uargs[1]])
                elif ulen == 1:
                    self.display_options()
            else:
                print("collector not initialized, use >>create collector] to initialize")

        elif uargs[0] in options['help']:
            print(help_string)

        elif uargs[0] in options['exit']:
            print("exiting...")
            self.close(uargs)

        else:
            print("unrecognized command {}".format(uargs[0]))

    def close(self, uargs):
        atexit.unregister(self.close)
        print("closing...")
        if self.Network:
            if "network_save_path" in self.Collecter.settings:
                self.Network.save(self.Collecter.settings["network_save_path"])
            elif len(uargs) > 1:
                self.Network.save(*uargs[1:])
            else:
                print("no save path, using default")
                self.Network.save("./Network_default")
        if self.Collecter:
            self.Collecter.close()
        self.running = False

    a = 9

    @staticmethod
    def table(a1, a2, a=None):
        if not a:
            a = CustomConsole.a

        if len(a1) > a * 2:
            a1 = a1[:a * 2 - 3] + "..."

        if len(a2) > a * 2:
            a2 = a2[:a * 2 - 3] + "..."

        l1 = len(a1)
        l2 = len(a2)
        p1 = ' ' * (round(a - l1 / 2))
        p2 = ' ' * (round(a - l2 / 2))

        print("|{}{}{}|{}{}{}|".format(p1, a1, p1, p2, a2, p2))
        print("-" * (4 * a + 2))

    def display_options(self):
        print("-" * (4 * self.a + 2))
        CustomConsole.table("NAME", "VALUE")
        for x in self.Collecter.settings:
            CustomConsole.table(x, self.Collecter.settings[x])
        if self.Network:
            if self.Network.built:
                print(self.Network.summary())


if __name__ == '__main__':
    
    settings = DataProcessor.load_processor("../API-metadata.xml", "../metadata-format.xml").settings

    CC = CustomConsole()
    
    with open("CLine_help.txt", "r") as CH:
        CC.help_string = CH.read()

    atexit.register(CC.close, [])
    print(settings)
    if "startuphook" in settings:
        CC.run(settings["startuphook"].split(CC.delim))

    while CC.running:
        line = input("Virus-API>> ")
        CC.run(line.split(CC.delim))
