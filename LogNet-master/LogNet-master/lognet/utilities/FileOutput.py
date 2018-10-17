import os
import errno
import pandas as pd
import csv

class FileOutput():
    """
    A class to write pandas dataframe to CSV with header in a separate text file
    """
    def __init__(self, output_file_path):

        self.header_written = False
        if not os.path.exists(output_file_path):
            os.mkdir(output_file_path)

        if os.path.isdir(output_file_path):
            self.output_path = output_file_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), output_file_path)

    # Do this only once
    def _writeHeader(self, header_list):

        print("Writing header file: headers.txt")
        with open(os.path.join(self.output_path, 'headers.txt'), 'w') as header_file:
            csv_writer = csv.writer(header_file, dialect='excel')
            csv_writer.writerow(header_list)
        self.header_written = True

    def write_csv(self, file_name, file_data):

        assert isinstance(file_data, pd.core.frame.DataFrame)==True
        if not self.header_written:
            self._writeHeader(file_data.columns.values.tolist())
        csv_filename = os.path.join(self.output_path, file_name)
        print(f"Writing CSV file: {csv_filename}")
        file_data.to_csv(csv_filename, index=False, header=None)

    def getFilePath(self):
        return self.output_path