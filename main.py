import os
import subprocess



class LlamaCommand():
    def __init__(self) -> None:
        self.filename  = ''
        self.sample    = ''
        self.profiling = ''

    def output_filename(self, settings) -> str:
        assert settings in list(range(6))

        post_descriptions = [
            "_only_describe",
            "_only_describe_with_profiling",
            "_answer_9_Qs",
            "_answer_9_Qs_with_profiling",
            "_answer_9_Qs_at_once",
            "_answer_9_Qs_at_once_with_profiling",
        ]
        return self.filename + post_descriptions[settings]

    def prompt(self, settings) -> str:
        assert settings in list(range(6))

        returning_prompt = \
"""\\begin{Instruction}
Answer the questions while using the context of the input.
The input is a random sample of the large dataset.
\\end{Instruction}

\\begin{Input}
""" + self.sample + """
\\end{Input}
"""

        if (settings+1) % 2 == 0:  # The odd settings are with profiling
            returning_prompt += \
"""
\\begin{InputB}
""" + self.profiling + """
\\end{InputB}
"""

        if settings < 2:  # The first two settings do not contain 9 questions
            returning_prompt += \
""""
\\begin{Question}
Please describe the dataset in sentences.
\\end{Question}

\\begin{Answer}
"""

        elif settings < 4:  # The 3rd and 4th settings contain 9 questions
            returning_prompt += \
"""
\\begin{Question}
Answer all the questions:
1. Can you describe the dataset in one sentence?
2. What does the dataset look like?
3. Can you group the headers?
4. What are the value types and value ranges for the most important headers?
5. Where is the data from?
6. In what format or in what way does the dataset mention time?
7. In what format or in what way does the dataset mention location?
8. Is there anything unclear about the data, or do you have reason to doubt the quality?
9. Is there anything that you can point out or analyse in more detail?
\\end{Question}

\\begin{Answer}
1. The dataset is"""

        elif settings < 6:  # The 5th and 6th settings contain 9 questions, but ask for answering together
            returning_prompt += \
"""
\\begin{Question}
Given these 9 aspects:
1. Can you describe the dataset in one sentence?
2. What does the dataset look like?
3. Can you group the headers?
4. What are the value types and value ranges for the most important headers?
5. Where is the data from?
6. In what format or in what way does the dataset mention time?
7. In what format or in what way does the dataset mention location?
8. Is there anything unclear about the data, or do you have reason to doubt the quality?
9. Is there anything that you can point out or analyse in more detail?
Describe the dataset in sentences by viewing these 9 aspects.
\\end{Question}

\\begin{Answer}
"""
        return returning_prompt

    def command(self, settings) -> list:
        assert settings in list(range(6))
        return [ r"./llama.cpp/bin/main", "--reverse-prompt", "\\end{Answer}",
                                          # "--interactive", "--color",
                                          "--ctx-size", "2048",
                                          "--n-predict", "-1",
                                          "--threads", "24",
                                          "--batch-size", "256",
                                          "--temp", "0.2",
                                          "--n-gpu-layers", "64",
                                          "--model", "./llama.cpp/models/llama-2-13b/ggml-model-f16.gguf",
                                          # "--model", "./llama.cpp/models/CodeLlama-34b/ggml-model-q5_1.gguf",
                                          # "--repeat_penalty", "1.2",
                                          # "--repeat-last-n", "-1",
                                          # "--no-penalize-nl",
                                          "--prompt", self.prompt(settings) ]


def load_sample_txt(filename):
    with open(filename) as txt:
        content = txt.read()
        content = content.replace('"\n"', '", "')
        content = content.split('", "')
        content[0] = content[0][1:]
        content[-1] = content[-1][:-2]
        filenames = content[::2]
        all_sample_and_profiling = content[1::2]
        return list(zip(filenames, all_sample_and_profiling))


def split_sample_and_profiling(sample_and_profiling: str):
    sample, profiling = sample_and_profiling.split("\n\n\n")
    sample    = sample.replace("Dataset sample: \n", '')
    profiling = profiling.replace("Column profiling: \n", '')
    profiling = profiling.replace("a http://schema.org/Float", "float")
    profiling = profiling.replace("a http://schema.org/Integer", "interger")
    profiling = profiling[:-1]
    return sample, profiling


def main():
    my_command = LlamaCommand()
    filename_sample_profiling = load_sample_txt("data/task1data/task1_sample.txt")

    os.makedirs("output", exist_ok=True)
    for filename, sample_profiling in filename_sample_profiling:
        sample, profiling = split_sample_and_profiling(sample_profiling)
        my_command.filename  = filename
        my_command.sample    = sample
        my_command.profiling = profiling

        # for settings in range(4, 6):
        for settings in range(6):
            with open(f"output/{my_command.output_filename(settings)}.txt", 'w') as output_file:
                subprocess.call(stdout=output_file, args=my_command.command(settings))



if __name__ == "__main__":
    main()