#!/usr/bin/env python

"""
@package mi.dataset.parser.dofst_p_wfp
@file mi-dataset/mi/dataset/parser/dofst_p_wfp.py
@author Samuel Dahlberg
@brief Parser for the dofst_p_wfp dataset driver.

This file contains code for the dofst_p_wfp parser and code to produce data
particles for the DOFST instrument data from the Prawler WFP instrument
(CTD + DOFST, with an optional FLORT). Files are mixed binary/ASCII.

Science section layout (record type 'E'):

    '@@@' <u16 len> <u16 len> 'E'           len (stored twice) and data type.
                                            Len being number of characters after this header
    '@@@' <u16 crc> <u16 len> 'E'           inner header; CRC-16/XMODEM and len
                                            This len being number of characters after this inner header
    column list (e.g. 'EP,CD,CT,CC,OT,O2,CH,TB[,C3]') \r\n
    Science records
    \r\n                                   section terminator

Science record:

    bytes 0-3   uint32 big-endian unix epoch  -- RAW BINARY
    byte  4     ','
    After that, comma-separated ASCII-hex fields, one per column-list entry after EP (epoch time)
    CD,CT,CC (CTD), OT,O2 (DOFST), CH,TB,C3 (FLORT, when applicable).
    Science Fields are printed '%04x' style: Fields are 4 char minimum but will increase up to 8.
    terminated by LF CR (\n\r), different order than the section terminator

Other fields 'G' (engineering) and 'S' (station-list) are skipped
"""

import binascii
import re
from struct import unpack_from

from mi.core.common import BaseEnum
from mi.core.exceptions import SampleException
from mi.core.instrument.dataset_data_particle import DataParticle, DataParticleKey
from mi.core.log import get_logger
from mi.dataset.dataset_parser import SimpleParser

log = get_logger()

# seconds between the NTP epoch (1900-01-01) and the unix epoch (1970-01-01)
NTP_EPOCH_OFFSET = 2208988800

EPOCH_LEN = 4                    # leading binary timestamp
HEX_FIELD_LEN = 4                # width of a settled ASCII-hex field
MIN_CTD_FIELDS = 3               # CD, CT, CC
SCIENCE_TYPE = b'E'
SECTION_TERMINATOR = b'\r\n'

# Regex for the start of a prawler data packet, and for a record terminator
# byte. Records end LF CR, but a lone LF or CR is tolerated; the terminator
# search must start after the binary epoch, which can contain either byte.
REC_REGEX = re.compile(br'Record\[(\d+)\]:@@@')
RECORD_END = re.compile(br'[\r\n]')


class DataParticleType(BaseEnum):
    DOFST_P = 'dofst_p_wfp_instrument'


class DofstPDataParticle(DataParticle):
    """
    Class for generating the dofst prawler instrument particle.
    """

    _data_particle_type = DataParticleType.DOFST_P

    def _build_parsed_values(self):
        """
        Build parsed values for Instrument Data Particle.
        @return: list containing type encoded "particle value id:value" dictionary pairs
        """

        return [{DataParticleKey.VALUE_ID: name, DataParticleKey.VALUE: value}
                for name, value in self.raw_data.items()]


class DofstPParticleKey(BaseEnum):
    """
    Class that defines fields that need to be extracted for the data particle.
    """

    TEMPERATURE = 'optode_temperature'
    DIS_OXG = 'estimated_oxygen_concentration'


class DofstPWfpParser(SimpleParser):
    """
    dofst_p_wfp parser.
    Will parse the dofst specific data from the generic prawler file.
    """

    @staticmethod
    def parse_hex_field(field):
        """
        Convert one ASCII-hex science field to an int.
        @return: the value, or None if the field is malformed or not the
        settled 4-char width (an oversized field is a stale buffer value,
        seen at the start of bursts; it is not a usable reading).
        """
        if len(field) != HEX_FIELD_LEN:
            return None
        try:
            return int(field, 16)
        except ValueError:
            return None

    def parse_record(self, data, here, section_end):
        """
        Parse one science record beginning at offset `here`. Store only the dofst_p instrument data
        @return: (particle_data, epoch_time, record_size).
        particle_data is None with a valid size when the record's values are
        unusable but its boundary was found (caller may skip just this
        record); all three are None when the record structure is
        unrecoverable (caller should abandon the section).
        """

        if here + EPOCH_LEN + 1 > section_end:
            return None, None, None

        epoch_time = unpack_from('>L', data, here)[0]

        if data[here + EPOCH_LEN:here + EPOCH_LEN + 1] != b',':
            return None, None, None

        # Find the record terminator, then size the record. Fields are
        # variable width, so scanning is the only reliable way to do this.
        match = RECORD_END.search(data, here + EPOCH_LEN + 1, section_end)
        if match is None:
            return None, None, None
        end = match.start()

        record_size = end - here + 1
        second = data[end + 1:end + 2]
        if second in (b'\r', b'\n') and second != data[end:end + 1]:
            record_size += 1                    # LF CR (or CR LF) pair

        fields = data[here + EPOCH_LEN + 1:end].split(b',')
        if len(fields) < MIN_CTD_FIELDS:
            return None, epoch_time, record_size

        opt_temp = self.parse_hex_field(fields[3])       # OT
        opt_o2 = self.parse_hex_field(fields[4])         # O2
        # fields[0:4] are CTDPF (CD, CT, CC); fields[5:8] are FLORT counts when
        # a FLORT is fitted. They are parsed by their own drivers.

        if opt_temp is None or opt_o2 is None:
            return None, epoch_time, record_size

        # Decimal precision provided by McLane
        particle_data = {
            DofstPParticleKey.TEMPERATURE: opt_temp / 1000.0,
            DofstPParticleKey.DIS_OXG: opt_o2 / 1000.0,
        }

        return particle_data, epoch_time, record_size

    def parse_file(self):
        """
        Parse a data record from the prawler data file, storing only the dofst_p instrument data
        """

        data = self._stream_handle.read()

        for record_match in REC_REGEX.finditer(data):
            start = record_match.end() - 3  # offset of '@@@'

            if len(data) < start + 8:
                break  # header cut off at EOF
            if data[start + 7:start + 8] != SCIENCE_TYPE:
                continue  # 'G', 'S'

            # Grabs the science data length, a value stored in the header.
            section_start = start + 8
            payload_len = unpack_from('>H', data, start + 3)[0]
            section_end = section_start + payload_len

            # Broken science data exception
            if section_end > len(data):
                self._exception_callback(SampleException(
                    'Science section at offset %d extends past end of file' % start))
                # We could parse what is there, but I think we want to avoid in case bad data
                # section_end = len(data)                  # parse what is there
                continue
            # If science header length does not end on an expected line break character...
            elif data[section_end:section_end + 2] != SECTION_TERMINATOR:
                self._exception_callback(SampleException(
                    'Science section at offset %d has an inconsistent length header' % start))
                continue
            # Check the crc cert to make sure the data natches what the instrument expects
            elif data[section_start:section_start + 3] == b'@@@':
                stored_crc, crc_len = unpack_from('>2H', data, section_start + 3)
                if binascii.crc_hqx(data[section_start + 8:section_end], 0) != stored_crc:
                    self._exception_callback(SampleException(
                        'Science section at offset %d failed its CRC check' % start))

            # skip the column list line
            columns_end = data.find(b'\r\n', section_start, section_end)
            if columns_end < 0:
                continue
            here = columns_end + 2

            # Parse and save each data record
            while here < section_end:
                particle_data, epoch_time, record_size = \
                    self.parse_record(data, here, section_end)

                if record_size is None:
                    log.error('Unrecoverable science record at offset %d', here)
                    self._exception_callback(SampleException(
                        'Unrecoverable science record at offset %d' % here))
                    break

                if particle_data is None:
                    log.error('Erroneous science record at offset %d', here)
                    self._exception_callback(SampleException(
                        'Erroneous science record at offset %d' % here))
                else:
                    particle = self._extract_sample(
                        self._particle_class, None, particle_data,
                        internal_timestamp=epoch_time + NTP_EPOCH_OFFSET)
                    if particle is not None:
                        self._record_buffer.append(particle)
                        log.trace('Parsed particle: %r', particle_data)

                here += record_size
