import os
import unittest

from mi.core.log import get_logger
from mi.dataset.dataset_driver import ParticleDataHandler
from mi.dataset.driver.vel3d_k.wfp_stc.resource import RESOURCE_PATH
from mi.dataset.driver.vel3d_k.wfp_stc.vel3d_k_wfp_stc_telemetered_driver import parse

__author__ = 'mworden'
log = get_logger()


class DriverTest(unittest.TestCase):

    def test_one(self):

        source_file_path = os.path.join(RESOURCE_PATH, 'A0000001_WithBeams.DEC')

        particle_data_handler = ParticleDataHandler()

        particle_data_handler = parse(None, source_file_path, particle_data_handler)

        log.debug("SAMPLES: %s", particle_data_handler._samples)
        log.debug("FAILURE: %s", particle_data_handler._failure)

        self.assertEquals(particle_data_handler._failure, False)


if __name__ == '__main__':
    test = DriverTest('test_one')
    test.test_one()
