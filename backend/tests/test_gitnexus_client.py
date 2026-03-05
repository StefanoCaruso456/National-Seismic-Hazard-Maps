import unittest
from unittest.mock import patch

from app.gitnexus_client import resolve_gitnexus_command


class GitNexusClientTests(unittest.TestCase):
    def test_resolve_prefers_global_gitnexus_binary(self) -> None:
        with patch("app.gitnexus_client.shutil.which") as which_mock:
            which_mock.side_effect = lambda binary: "/usr/local/bin/gitnexus" if binary == "gitnexus" else None
            resolved = resolve_gitnexus_command("npx -y gitnexus@latest mcp")
        self.assertEqual(resolved, ["gitnexus", "mcp"])

    def test_resolve_splits_custom_command(self) -> None:
        resolved = resolve_gitnexus_command("gitnexus mcp")
        self.assertEqual(resolved, ["gitnexus", "mcp"])


if __name__ == "__main__":
    unittest.main()
