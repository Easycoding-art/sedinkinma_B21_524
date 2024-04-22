from generate_alphabet import create_data
from recognizer import test
import os

def start_experiment(letters, font, lang_name, test_number) :
    if os.path.exists(lang_name):
        print("Experiment data have already created!")
    else:
        create_data(font, letters, lang_name, test_number)
    if os.path.exists('test_results.csv'):
        print("Experiment results have already got!")
    else:
        test('test_cases.csv', 'test_cases', f"{lang_name}.csv")

if __name__ == '__main__' :
    letters = list('ءآأؤإئاةتثجحخدذرزسشصضطظعغ')
    start_experiment(letters, 'c:\WINDOWS\Fonts\ARIAL.TTF', 'Arabian', 20)