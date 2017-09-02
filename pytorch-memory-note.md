# Notes about Memory (GPU) related issues in PyTorch

- Since PyTorch has a caching allocator, even though your variables are **deleted** they wont reflect in (device) memory (nvidia-smi or any other way you monitor). But the memory **will not** go up if you create a new variable or anything, the cache will be used.
- But this isn't applicable to multiple processes as each has its own. [Ref](https://discuss.pytorch.org/t/how-to-clear-some-gpu-memory/1945/4)
- One more note for Training/Testing iterations: During Inference, be sure to make your input `Variable(s)` `volatile=True`. This will remove any abnormal extra memory usage during testing stage
- Above two mostly werent familiar to me. Other common things, many face problems with `multiprocessing` and `DataLoaders` if you kill process [abruptly](https://github.com/pytorch/pytorch/issues/1085#issuecomment-289160007)
- Will update once I encounter more (hopfeully not :P)
