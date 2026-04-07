# Copyright 2026 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import unittest

from source_resegmenter import refiner


class TestRefiners(unittest.TestCase):
    def test_labse(self):
        self._do_test_refiner("xlr-labse")

    def test_simalign(self):
        self._do_test_refiner("xlr-simalign")

    def _do_test_refiner(self, refiner_name):
        reference = "\n".join([
            "A New York, sono a capo di un'associazione no profit, chiamata Robin Hood.",
            "Quando non combatto la povertà, combatto gli incendi come assistente capitano di "
            "una brigata di pompieri volontari.",
            "Nella nostra citta, in cui i volontari supplementano i professionisti altamente "
            "qualificati, bisogna arrivare sulla luogo dell'incendio piuttosto in fretta se "
            "si vuole combinare qualcosa."])
        source_input = "\n".join([
            "Back in New York, I am the head of development for a non-profit called Robin Hood. ",
            "When I'm not fighting poverty, I'm fighting fires as the assistant captain of a "
            "volunteer fire company. Now ",
            "in our town, where the volunteers supplement a highly skilled career staff, you have "
            "to get to the fire scene pretty early to get in on any action. "])
        expected_output = "\n".join([
            "Back in New York, I am the head of development for a non-profit called Robin Hood.",
            "When I'm not fighting poverty, I'm fighting fires as the assistant captain of a "
            "volunteer fire company.",
            "Now in our town, where the volunteers supplement a highly skilled career staff, you "
            "have to get to the fire scene pretty early to get in on any action."])
        if refiner_name == "xlr-labse":
            refined_source = refiner.xlr_labse(source_input, reference)
        elif refiner_name == "xlr-simalign":
            refined_source = refiner.xlr_simalign(source_input, reference, "en", "it")
        else:
            self.fail()
        self.assertEqual(refined_source, expected_output)


if __name__ == '__main__':
    unittest.main()
