# import json
# from API.ApiMain import DataProcessor

TEST_PATH = "C:/Users/vsaig/OneDrive/Documents/Science Fair/phys_geo/"
TEST_FILENAME = "phys_geo.csv"
strange_test_string = \
        '{'\
            '"format": "xlsx",'\
            '"data": {'\
                '"data_range": [ 2, 1, 164, 17 ],'\
                '"row_id_col": 0,'\
                '"col_id_row": 0,'\
                '"cols_used": [ 5, 15, 17 ],'\
                '"rows_unused": [],'\
                '"rename_cols": {'\
                    '"pdenpavg": "population_density",'\
                    '"elev": "elevation",'\
                    '"cen_c": "dist_to_coast"'\
                '}'\
            '}'\
        '}'

# with open(TEST_PATH + "fields.json") as test_fields:
#     with open(TEST_PATH + TEST_FILENAME) as test_data:
#         json_fields = json.load(test_fields)
#         dp = DataProcessor()
#         gd = dp.get_data_from_dir(TEST_PATH)
#         [print(x) for x in gd]
#         print("|\n" * 10)
#         test_2 = dp.get_data_from_dir_file_data("C:/Users/vsaig/OneDrive/Documents/Science Fair/Population Over 65/")[0]
#         print('\n'.join([str(x) for x in test_2]))
if __name__ == "__main__":
        print("idk")
        f = open("API/CLine_help.txt")
