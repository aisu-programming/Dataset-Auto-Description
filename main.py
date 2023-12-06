import os
import subprocess



class LlamaCommand():
    def __init__(self) -> None:
        self.filename             = ''
        self.sample               = ''
        self.sample_with_profiler = ''

    def output_filename(self, settings) -> str:
        assert settings in list(range(3))
        return self.filename + f"_{settings+1}"

    def prompt(self, settings) -> str:
        assert settings in list(range(3))

        if settings == 0:
            return \
"""Instruction:
Answer the questions while using the input and context.
The input includes dataset title, hearders, a random sample, and profiler result of the large dataset.

Input:
""" + self.sample + """
Question:
Describe the dataset in sentences.

Answer:
The dataset"""

        elif settings == 1:
            return \
"""Instruction:
Answer the questions while using the input and context.
The input includes dataset title, hearders and a random sample of the large dataset.

Input:
""" + self.sample_with_profiler + """
Question:
Describe the dataset in sentences.

Answer:
The dataset"""

        elif settings == 2:
            return \
"""Instruction:
Answer the questions while using the input and context.
The input includes dataset title, hearders, a random sample, and profiler result of the large dataset. 

Input:
""" + self.sample_with_profiler + """
Question:
Considering the following nine aspects:
1. Describe the dataset in one sentence?
2. What does the dataset look like?
3. Can you group the headers?
4. What are the value types and value ranges for the most important headers?
5. Where is the data from?
6. In what format or in what way does the dataset mention time?
7. In what format or in what way does the dataset mention location?
8. Is there anything unclear about the data, or do you have reason to doubt the quality?
9. Is there anything that you can point out or analyse in more detail?
Describe the dataset answering the nine aspects above in one compelete and coherent paragraph.

Answer:
The dataset"""


    def command(self, settings) -> list:
        assert settings in list(range(3))
        return [ r"./llama.cpp/bin/main", "--reverse-prompt", "\n",
                                          # "--interactive", "--color",
                                          "--ctx-size", "2048",
                                          "--n-predict", "-1",
                                          "--threads", "24",
                                          "--batch-size", "256",
                                          "--temp", "0.2",
                                          "--n-gpu-layers", "64",
                                          # "--model", "./llama.cpp/models/llama-2-7b/ggml-model-f16.gguf",
                                          # "--model", "./llama.cpp/models/llama-2-13b/ggml-model-f16.gguf",
                                          "--model", "./llama.cpp/models/CodeLlama-34b/ggml-model-q5_1.gguf",
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


def main():
    my_command = LlamaCommand()
    filename_sample          = load_sample_txt("data/task1data/task1_sample.txt")
    filename_sample_profiler = load_sample_txt("data/task1data/task1_sample_profiler.txt")

    os.makedirs("output/task1_llama34b", exist_ok=True)
    for (filename, sample), (_, sample_with_profiler) in zip(filename_sample, filename_sample_profiler):
        # sample, profiling = split_sample_and_profiling(sample_profiling)
        my_command.filename             = filename
        my_command.sample               = sample
        my_command.sample_with_profiler = sample_with_profiler

        for settings in range(3):
            with open(f"output/task1_llama34b/{my_command.output_filename(settings)}.txt", 'w') as log:
                subprocess.call(stdout=log, args=my_command.command(settings))

            with open(f"output/task1_llama34b_p{settings+1}.txt", 'a') as description_file:
                with open(f"output/task1_llama34b/{my_command.output_filename(settings)}.txt", 'r') as log:
                    description = log.read().split("Answer:\n")[1][:-1]
                    description_file.write(f'{filename}\t{description}\n')



if __name__ == "__main__":
    main()