import unittest

from csdemo_mapextractor.entity_data import parse_states


class EntityDataTests(unittest.TestCase):
    def test_parses_source2viewer_disabled_predicate(self) -> None:
        text = """
        m_entityKeyValues = [
          { keyValues3Data = { values =
            {
              classname = "func_brush"
              origin = [ 1.0, 2.0, 3.0 ]
              startdisabled = true
            }
          } }
          { keyValues3Data = { values =
            {
              classname = "func_brush"
              origin = [ 4.0, 5.0, 6.0 ]
              enabled = false
            }
          } }
          { keyValues3Data = { values =
            {
              classname = "func_brush"
              origin = [ 7.0, 8.0, 9.0 ]
            }
          } }
        ]
        """

        states = parse_states(text, "func_brush")

        self.assertEqual([True, True, False], [state.disabled for state in states])


if __name__ == "__main__":
    unittest.main()