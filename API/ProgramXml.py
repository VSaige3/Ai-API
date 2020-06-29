from xml.etree import ElementTree as ET
from typing import Optional
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



