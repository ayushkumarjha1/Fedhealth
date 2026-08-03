"""
Unit Tests for FedHealth CLI interface.
"""

import unittest
import sys
from unittest.mock import patch

from fedpro.cli.main import main

class TestCLI(unittest.TestCase):
    def test_cli_help(self):
        """Verify CLI prints help and exits without error when no arguments provided."""
        with patch.object(sys, 'argv', ['fedhealth']):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)

if __name__ == "__main__":
    unittest.main()
