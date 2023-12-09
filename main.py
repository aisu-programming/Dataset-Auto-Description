import os
import subprocess



class LlamaCommand():
    def __init__(self, model_id, filename, sample, sample_with_profiler) -> None:
        assert model_id in list(range(3))
        self.model_path = [
            "./llama.cpp/models/llama-2-7b/ggml-model-f16.gguf",
            "./llama.cpp/models/llama-2-13b/ggml-model-f16.gguf",
            "./llama.cpp/models/CodeLlama-34b/ggml-model-q5_1.gguf",
        ][model_id]
        self.filename             = filename
        self.sample               = sample
        self.sample_with_profiler = sample_with_profiler

    def output_filename(self, prompt_id) -> str:
        assert prompt_id in list(range(3))
        return self.filename + f"_{prompt_id+1}"

    def prompt(self, prompt_id) -> str:
        assert prompt_id in list(range(3))

        if prompt_id == 0:
            return \
"""Instruction:
Answer the questions while using the input and context.
The input includes dataset title, headers and a random sample of the large dataset.

Input:
""" + self.sample + """
Question:
Describe the dataset in one complete and coherent paragraph.

Answer:
The """

        elif prompt_id == 1:
            return \
"""Instruction:
Answer the questions while using the input and context.
The input includes dataset title, headers, a random sample, and profiler result of the large dataset.

Input:
""" + self.sample_with_profiler + """
Question:
Describe the dataset in one complete and coherent paragraph.

Answer:
The """

        elif prompt_id == 2:
            return \
"""Instruction:
Answer the questions while using the input and context.
The input includes dataset title, headers, a random sample, and profiler result of the large dataset. 

Input:
""" + self.sample_with_profiler + """
Question:
Considering the following eight aspects:
1. Summarize the dataset's overall content and purpose in one sentence.
2. Offer an overview of its structure, including data types and general organization.
3. Identify and group the dataset's headers, explaining their relevance and interrelationships.
4. Detail the value types and range for key headers, emphasizing significant trends or patterns.
5. Describe how the dataset represents time, including the format and temporal scope.
6. Explain the representation of location in the dataset, noting specific formats or geospatial details.
7. Discuss any ambiguities or quality concerns in the data, providing examples where necessary.
8. Highlight any notable findings or potential areas for deeper analysis, based on your initial review.
Describe the dataset covering the eight aspects above in one complete and coherent paragraph.

Answer:
The """


    def command(self, prompt_id) -> list:
        assert prompt_id in list(range(3))
        return [ r"./llama.cpp/bin/main", "--reverse-prompt", "\n",
                                          # "--interactive", "--color",
                                          "--ctx-size", "2048",
                                          "--n-predict", "-1",
                                          "--threads", "24",
                                          "--batch-size", "256",
                                          "--temp", "0.2",
                                          "--n-gpu-layers", "64",
                                          "--model", self.model_path,
                                          # "--repeat_penalty", "1.2",
                                          # "--repeat-last-n", "-1",
                                          # "--no-penalize-nl",
                                          "--prompt", self.prompt(prompt_id) ]


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

    for task_id in range(2):
        filename_sample          = load_sample_txt(f"data/task{task_id+1}_sample.txt")
        filename_sample_profiler = load_sample_txt(f"data/task{task_id+1}_sample_profiler.txt")

        for model_id in range(3):
            saving_dir  = "output/"
            saving_dir += [ f"task{task_id+1}_llama7b", f"task{task_id+1}_llama13b", f"task{task_id+1}_llama34b" ][model_id]
            os.makedirs(saving_dir, exist_ok=True)
            
            for (filename, sample), (_, sample_with_profiler) in zip(filename_sample, filename_sample_profiler):
                my_command = LlamaCommand(model_id, filename, sample, sample_with_profiler)

                for prompt_id in range(3):
                    with open(f"{saving_dir}/{my_command.output_filename(prompt_id)}.txt", 'w') as log:
                        subprocess.call(stdout=log, args=my_command.command(prompt_id))

                    with open(f"{saving_dir}_p{prompt_id+1}.txt", 'a') as description_file:
                        with open(f"{saving_dir}/{my_command.output_filename(prompt_id)}.txt", 'r') as log:
                            description = log.read().split("Answer:\n")[1][:-1]
                            description_file.write(f'{filename}\t{description}\n')



if __name__ == "__main__":
    main()