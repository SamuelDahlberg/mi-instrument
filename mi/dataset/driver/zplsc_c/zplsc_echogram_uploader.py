#!/usr/bin/env python

"""
@package mi.dataset.driver.zplsc_c
@file mi-dataset/mi/dataset/driver/zplsc_c/zplsc_echogram_uploader.py
@author Mark Steiner
@brief Driver to upload metadata for pre-generated zplsc echograms into uFrame

Release notes:

Initial Release
"""

import os
import re
import netCDF4
from datetime import datetime, timedelta

from mi.core.versioning import version
from mi.core.instrument.dataset_data_particle import DataParticle, DataParticleKey, DataParticleValue
from mi.core.common import BaseEnum
from mi.core.exceptions import SampleException
from mi.core.time_tools import time_to_ntp_date_time, string_to_ntp_date_time
from mi.core.log import get_logger

from mi.dataset.dataset_driver import DataSetDriver, SimpleDatasetDriver
from mi.dataset.dataset_driver import ParticleDataHandler, ProcessingInfoKey

from mi.dataset.dataset_parser import SimpleParser
from mi.dataset.dataset_parser import DataSetDriverConfigKeys


log = get_logger()

HOURLY_FILENAME_DATE_FORMAT = 'OOI-D%Y%m%d-T%H%M%S.nc'


@version("0.1.0")
def parse(unused, echogram_file_path, particle_data_handler):
    """
    This is the method called by uFrame
    :param unused
    :param echogram_file_path This is the full path and filename of the echogram to be uploaded
    :param particle_data_handler - Object to hold the output of the parser
    :return particle_data_handler
    """

    # Pass the echogram_file_path all the way down to the parser so it can be set on the
    # generated zplsc_metadata particle
    driver = ZplscEchogramUploadDriver(unused,
                                        None,
                                        particle_data_handler,
                                        echogram_file_path)
    #  Parse the file and populate the particle_data_handler with particles
    driver.processFileStream()

    # Transfer provenance information from the parser to the particle_data_handler
    driver.add_provenance_to_processing_info()

    return particle_data_handler


class ZplscProvenanceKey(BaseEnum):
    """
    Class that defines fields that need to be extracted from the
    "Provenance" group from the co-located component netCDF file.
    These fields describe how the echogram was generated.
    """
    DATA_FILE_NAME = "data_file_name"
    CONVERSION_SOFTWARE_NAME = "conversion_software_name"
    CONVERSION_SOFTWARE_VERSION = "conversion_software_version"
    CONVERSION_TIME = "conversion_time"


class ZplscMetadataParticleKey(BaseEnum):
    """
    Class that defines fields that need to be extracted from the data
    """
    FILE_TIME = "zplsc_timestamp"   # Timestamp of first record in raw data file
    ECHOGRAM_PATH = "filepath"      # Full path of echogram to be uploaded


class ZplscEchogramType(BaseEnum):
    """
    Class that defines the 3 types of echogram files that we are expecting
    """
    AVERAGED = "Averaged"
    FULL = "Full"
    HOURLY = "Hourly"


class ZplscMetadataParticle(DataParticle):
    """
    Class for generating the zplsc_metadata data particle. Note that this is the
    same stream as the the metadata particle generated by the zplsc_b parser.
    """
    _data_particle_type = "zplsc_metadata"

    def __init__(self, raw_data,
                 port_timestamp=None,
                 internal_timestamp=None,
                 preferred_timestamp=DataParticleKey.INTERNAL_TIMESTAMP,
                 quality_flag=DataParticleValue.OK,
                 new_sequence=None):
        super(ZplscMetadataParticle, self).__init__(raw_data,
                                                    port_timestamp,
                                                    internal_timestamp,
                                                    preferred_timestamp,
                                                    quality_flag,
                                                    new_sequence)

    def _build_parsed_values(self):
        """
        The path to the echogram is the only field we need to set (besides the timestamps)
        """
        data_list = []
        data_list.append(self._encode_value(ZplscMetadataParticleKey.ECHOGRAM_PATH,
                                            self.raw_data[ZplscMetadataParticleKey.ECHOGRAM_PATH], str))
        data_list.append(self._encode_value(ZplscMetadataParticleKey.FILE_TIME,
                                            self.raw_data[ZplscMetadataParticleKey.FILE_TIME], float))
        return data_list


class ZplscEchogramUploadDriver(SimpleDatasetDriver):
    """
    Derived class to instantiate the actual file parser
    """
    def __init__(self, unused, stream_handle, particle_data_handler, echogram_filepath):
        self._echogram_filepath = echogram_filepath
        super(ZplscEchogramUploadDriver, self).__init__(unused, stream_handle, particle_data_handler)

    def _build_parser(self, stream_handle):
        parser_config = {
            DataSetDriverConfigKeys.PARTICLE_MODULE: 'mi.dataset.driver.zplsc_c.zplsc_echogram_uploader',
            DataSetDriverConfigKeys.PARTICLE_CLASS: 'ZplscMetadataParticle'
        }

        parser = ZplscEchogramUploadParser(parser_config, stream_handle,
                                           self._exception_callback, self._echogram_filepath)
        return parser

    def add_provenance_to_processing_info(self):
        # Transfer provenance information from the parser to the particle_data_handler.
        # Translate the echogram generator-specific provenance keys extracted by the parser into
        # generic processing information that is understood by the caller and the rest of the system.

        if ZplscProvenanceKey.DATA_FILE_NAME in self._parser.provenance.keys():
            self._particle_data_handler.setProcessingInfo(ProcessingInfoKey.DATA_FILE,
                        self._parser.provenance[ZplscProvenanceKey.DATA_FILE_NAME])

        if ZplscProvenanceKey.CONVERSION_SOFTWARE_NAME in self._parser.provenance.keys():
            self._particle_data_handler.setProcessingInfo(ProcessingInfoKey.PARSER,
                        self._parser.provenance[ZplscProvenanceKey.CONVERSION_SOFTWARE_NAME])

        if ZplscProvenanceKey.CONVERSION_SOFTWARE_VERSION in self._parser.provenance.keys():
            self._particle_data_handler.setProcessingInfo(ProcessingInfoKey.PARSER_VERSION,
                        self._parser.provenance[ZplscProvenanceKey.CONVERSION_SOFTWARE_VERSION])


class ZplscEchogramUploadParser(SimpleParser):
    def __init__(self,
                 config,
                 stream_handle,
                 exception_callback,
                 echogram_filepath):

        super(ZplscEchogramUploadParser, self).__init__(config, stream_handle, exception_callback)

        # The provenance that describes how the echogram was actually generated
        # will be extracted from an co-located "hourly" .nc file.
        self.provenance = {}

        self._echogram_filepath = echogram_filepath

    """
    Parser for zplsc_metadata.
    """
    def parse_file(self):

        echogram_dirname, echogram_filename = os.path.split(self._echogram_filepath)

        # The name of the echogram_filename can be of the "Averaged", "Full"
        # or simple "Hourly" formats
        # Examples:
        #     CE07SHSM_Bioacoustic_Echogram_20191020-20191027_Calibrated_Sv_Averaged.nc
        #     CE07SHSM_Bioacoustic_Echogram_20191020-20191027_Calibrated_Sv_Full_20191020.nc
        #     OOI-20191020-T013835.nc
        averaged_or_full_filename_regex_str = \
            r'Bioacoustic_Echogram_(?P<start_date>[0-9]{8})-(?P<stop_date>[0-9]{8})_Calibrated_Sv_(?P<type>Averaged|Full)_?(?P<date>[0-9]{8})?\.nc'
        hourly_filename_regex_str = r'OOI-D[0-9]{8}-T[0-9]{6}\.nc'

        m = re.compile(averaged_or_full_filename_regex_str).search(echogram_filename)
        if m:
            echogram_type = m.group('type')
        else:
            m = re.compile(hourly_filename_regex_str).match(echogram_filename)
            if m:
                echogram_type = ZplscEchogramType.HOURLY
            else:
                error_msg = "Filename \"%s\" not in either of the expected formats: \"%s\" or \"%s\"" % \
                        (echogram_filename, averaged_or_full_filename_regex_str, hourly_filename_regex_str)
                log.error(error_msg)
                raise SampleException(error_msg)

        # Use the regex match captures from the echogram_filename to generate another regex as well as
        # start date and end date criteria to find the hourly .nc files that correspond to the echogram.
        if echogram_type == ZplscEchogramType.FULL:
            hourly_file_regex = 'OOI-D' + m.group('date') + r'-T[0-9]{6}\.nc'
            start_day_datetime = datetime.strptime(m.group('date'), "%Y%m%d")
            # Assume the stop time to be the beginning of the next day
            stop_day_datetime = start_day_datetime + timedelta(days=1)
        elif echogram_type == ZplscEchogramType.AVERAGED:
            hourly_file_regex = r'OOI-D[0-9]{8}-T[0-9]{6}\.nc'
            start_day_datetime = datetime.strptime(m.group('start_date'), "%Y%m%d")
            # The stop time in the filename is already the beginning of the next day
            stop_day_datetime = datetime.strptime(m.group('stop_date'), "%Y%m%d")
        else: # ZplscEchogramType.HOURLY
            hourly_file_regex = echogram_filename
            start_day_datetime = datetime.strptime(echogram_filename, HOURLY_FILENAME_DATE_FORMAT)
            stop_day_datetime = start_day_datetime

        # Get a list of the hourly files that correspond to the echogram time range.
        # These hourly files contain the provenance for the echogram.
        hourly_files = [f for f in os.listdir(echogram_dirname)
                        if re.match(hourly_file_regex, f)
                            and (echogram_type == ZplscEchogramType.HOURLY
                                 or
                                 (start_day_datetime <= datetime.strptime(f, HOURLY_FILENAME_DATE_FORMAT) < stop_day_datetime))]

        if len(hourly_files) == 0:
            error_msg = "Hourly files from %s to %s corresponding to \"%s\" echogram \"%s\" could not be found that match regex \"%s\"" %\
                 (start_day_datetime.strftime(HOURLY_FILENAME_DATE_FORMAT), stop_day_datetime.strftime(HOURLY_FILENAME_DATE_FORMAT),
                  echogram_type, echogram_filename, hourly_file_regex)
            log.error(error_msg)
            raise SampleException(error_msg)

        particle_data_dict = {}

        # Use only the first hourly file in the sorted list to get the provenance information.
        # Get the provenance before we generate the particle since we use the
        # Provenance.conversion_time to set the DataParticle.driver_timestamp
        self.set_provenance_from_hourly_file(os.path.join(echogram_dirname, sorted(hourly_files)[0]))
        self.modify_provenance_for_echogram_type(echogram_type, start_day_datetime, stop_day_datetime)

        # The HOURLY file and the FULL file will have the same first ping_time which we use as the
        # 'time' value in the cassandra record. Stream engine would filter out one of the records thinking
        # it is a dupe so we offset one of the timestamps by a small amount to prevent this from happening.
        timestamp_offset = 0.001 if echogram_type == ZplscEchogramType.FULL else 0

        # Add the internal_timestamp, zplsc_timestamp and the echogram path to the data dictionary
        # that will later be used to generate the zplsc_metadata particle
        first_ping_time = self.get_first_ping_time_from_echogram(echogram_type)
        particle_data_dict[DataParticleKey.INTERNAL_TIMESTAMP] = first_ping_time + timestamp_offset
        particle_data_dict[ZplscMetadataParticleKey.FILE_TIME] = first_ping_time
        particle_data_dict[ZplscMetadataParticleKey.ECHOGRAM_PATH] = self._echogram_filepath

        # Instantiate a ZplscMetadataParticle and populate it from the file_metadata_dict
        particle = self._extract_sample(ZplscMetadataParticle, None, particle_data_dict, None,
                                        particle_data_dict[DataParticleKey.INTERNAL_TIMESTAMP])

        # This field needs to be set after particle creation to overwrite the value set by the constructor
        particle.contents[DataParticleKey.DRIVER_TIMESTAMP] =\
            string_to_ntp_date_time(self.provenance[ZplscProvenanceKey.CONVERSION_TIME])

        if particle is not None and not particle.get_encoding_errors():
            self._record_buffer.append(particle)

    def set_provenance_from_hourly_file(self, hourly_file):
        nc4_dataset = netCDF4.Dataset(hourly_file)
        self.provenance[ZplscProvenanceKey.DATA_FILE_NAME] = nc4_dataset.groups['Provenance'].src_filenames
        self.provenance[ZplscProvenanceKey.CONVERSION_SOFTWARE_NAME] = nc4_dataset.groups['Provenance'].conversion_software_name
        self.provenance[ZplscProvenanceKey.CONVERSION_SOFTWARE_VERSION] = nc4_dataset.groups['Provenance'].conversion_software_version
        self.provenance[ZplscProvenanceKey.CONVERSION_TIME] = nc4_dataset.groups['Provenance'].conversion_time
        nc4_dataset.close()

    def modify_provenance_for_echogram_type(self, echogram_type, first_datetime, last_datetime):
        data_file_name = self.provenance.get(ZplscProvenanceKey.DATA_FILE_NAME, None)

        # Hourly files are generated from one raw data file so a range of files
        # does not need to be set in the provenance.
        if echogram_type == ZplscEchogramType.HOURLY or not data_file_name:
            return

        # Example of data file name: /data/testing/zplsc/ce04osps/2017/09/10/OOI-D20170910-T013835.raw
        daily_dirname, data_filename = os.path.split(data_file_name)
        data_filename_wo_ext, data_filename_ext = os.path.splitext(data_filename)

        if echogram_type == ZplscEchogramType.FULL:
            # Echogram is for a full day so use all times (wildcard) for that specific day
            self.provenance[ZplscProvenanceKey.DATA_FILE_NAME] = \
                data_file_name[:data_file_name.rfind('T')] + 'T*' + data_filename_ext
        else:
            # Echogram is the Average across all days and times in the specific date range
            # Don't include the stop day (ie. the next day)
            last_datetime = last_datetime - timedelta(days=1)
            base_time_range_str = data_filename_wo_ext[:data_filename_wo_ext.rfind('D')] + 'D*-T*' + data_filename_ext
            monthly_dirname = os.path.split(daily_dirname)[0]
            yearly_dirname = os.path.split(monthly_dirname)[0]

            self.provenance[ZplscProvenanceKey.DATA_FILE_NAME] = \
                os.path.join(yearly_dirname,
                             first_datetime.strftime('%m'),
                             first_datetime.strftime('%d'),
                             base_time_range_str)

            if last_datetime > first_datetime:
                self.provenance[ZplscProvenanceKey.DATA_FILE_NAME] = \
                    self.provenance[ZplscProvenanceKey.DATA_FILE_NAME] + ' ... ' + \
                    os.path.join(yearly_dirname,
                                 last_datetime.strftime('%m'),
                                 last_datetime.strftime('%d'),
                                 base_time_range_str)

    def get_first_ping_time_from_echogram(self, echogram_type):
        nc4_dataset = netCDF4.Dataset(self._echogram_filepath)
        if echogram_type == ZplscEchogramType.HOURLY:
            first_ping_time = nc4_dataset.groups['Beam'].variables['ping_time'][0]
        else: # AVERAGED and FULL echograms are not of nc4 format and don't have groups
            first_ping_time = time_to_ntp_date_time(nc4_dataset.variables['ping_time'][0])
        nc4_dataset.close()
        return first_ping_time
