import os
import errno
import lasio
import pandas as pd 

_HEADERS_TO_COPY = ['Version', 'Well', 'Parameter']

class LASOutput(object):
    r"""Class for writing LAS files.

    `LASOutput` is a class for writing LAS files. This class copies headers
    from a given source file and writes a given dataframe as curves.

    Arguments:
        output_path (string): Path where LAS file is written.
    """

    def __init__(self, output_path: str) -> None: 

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if os.path.isdir(output_path):
            self.output_path = output_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), output_path)

    def write_las(self, processed_df: pd.core.frame.DataFrame, source: lasio.las.LASFile, filename: str, location=None) -> None:
        r"""Write LAS file

        Arguments:
            processed_df: Dataframe that needs to be written
            source: LAS file from which headers are copied
            filename: Name of the output file
            location (tuple, optional): Latitude and longitude that is written to the header
        """

        if processed_df.empty:
            #TODO: This needs to be a warning through logging
            print('Input dataframe is empty')
            return 
 
        # Create new LAS file here
        processed_las = lasio.LASFile()

        # Copy the headers that haven't been changed
        for entry in _HEADERS_TO_COPY:
            processed_las.header[entry] = source.header[entry]

        # Insert location information to the header
        if location:
            assert(len(location) == 2)

            latitude = location[0]
            longitude = location[1] 

            processed_las.well['SLAT'] = lasio.HeaderItem('SLAT', unit='WGS84', value=latitude, descr='Surface Latitude')
            processed_las.well['SLON'] = lasio.HeaderItem('SLON', unit='WGS84', value=longitude, descr='Surface Longitude')

        # Insert curves now
        for entry in processed_df.columns:
            if entry == 'DEPT':
                processed_las.add_curve('DEPT', processed_df['DEPT'].values, unit='ft')
            else:
                processed_las.add_curve(entry, processed_df[entry].values)

        processed_las.write(os.path.join(self.output_path, filename), version=2)
