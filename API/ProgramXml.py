from os import PathLike
from xml.etree import ElementTree as ET
from typing import Optional, List, Union


class XMLReader:
    _format: str
    required = [
        "(TYPE)", "(METADATA_DPATH)", "(FILEDATA_DPATH)"
    ]

    def __init__(self, format: Optional[str]):
        if not format:
            format = open("../metadata-format.xml").read()
        _format = format

    def __call__(self, xml):
        root = ET.parse(xml)
        expected = ET.fromstring(self._format)
        return XMLReader._parse(root, expected)

    @staticmethod
    def _parse(root, expected):
        for child in expected:
            print(child)
            exp = root.find(child)
            if exp:
                return XMLReader._parse(root, exp)
            else:
                # something
                print(exp)

    @staticmethod
    def get_vals(data: Union[ET.ElementTree, ET.Element, str],
                 fmat: Union[ET.ElementTree, ET.Element, str],
                 names: Union[str, List[str]]) -> List[Union[ET.Element, str]]:
        # go through to find matching stuff
        if isinstance(names, str):
            names = [names]
        if isinstance(data, str):
            data = ET.parse(data)
        if isinstance(fmat, str):
            fmat = ET.parse(fmat)
        rets = []

        format_i = fmat.iter()
        data_i = data.iter()
        z = zip(format_i, data_i)
        for x in z:
            # print(x.tag)
            # f = match(data, x.tag)
            # if f:
            #     print(f"{x.tag}, {f.tag}!")

            if x[0].text in names:
                # names.remove(x[0].text)
                #     rets.append(*get_vals(x, f, names))
                a = x[1].text
            elif x[0].tag in names:
                # names.remove(x[0].tag)
                #     rets.append(*get_vals(x, f, names))
                a = x[1]
            else:
                a = None
                continue
            if a is not None:
                rets.append(a)
        return rets
