import tempfile
import unittest
from pathlib import Path
from recovery import Candidate, Experience, Memory


class RecoveryTests(unittest.TestCase):
    def test_failure_changes_ranking_and_memory_is_class_specific(self):
        candidates = [Candidate('arm', 10, .7, True), Candidate('torso', 30, .6, True)]
        memory = Memory()
        self.assertEqual(memory.rank('figure', candidates, 40)[0]['region'], 'arm')
        memory.add(Experience('figure', 'arm', False))
        self.assertEqual(memory.rank('figure', candidates, 40)[0]['region'], 'torso')
        self.assertEqual(memory.rank('other', candidates, 40)[0]['region'], 'arm')

    def test_feasibility_and_persistence(self):
        memory = Memory([Experience('figure', 'torso', True)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.json'
            memory.save(path)
            restored = Memory.load(path)
            self.assertEqual(restored.experiences, memory.experiences)
        candidates = [Candidate('wide', 50, .9, True), Candidate('unreachable', 10, .9, False)]
        self.assertEqual(memory.rank('figure', candidates, 40), [])

    def test_current_failure_penalty(self):
        candidates = [Candidate('arm', 10, .7, True), Candidate('torso', 30, .6, True)]
        self.assertEqual(Memory().rank('figure', candidates, 40, 'arm')[0]['region'], 'torso')


if __name__ == '__main__':
    unittest.main()
