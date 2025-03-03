# MKG_Multi_Agent

This is a multi-agent system for a knowledge graph.

### In-progress troubleshooting
```
Let's take a step back here and think carefully about the goal of this pipeline you are helping me make. the `conv_text` being sent is a value in the json file @conversations_list.json (right now it only has one key-value because the input to @process_conversations.py only had one file @Saffir.json  ). I simply wish to use an LLM to infer X number of research requests from each conv_text passed. the idea is that the AI whose conversation is recorded in json in the input @Saffir.json can contain requests for information so we are now writing code to extract information requests from the conversation log. It is X research requests because we don't necessarily know how many should be extracted, therefore i am hard-coding 5 for now. what other means could we go about doing this, for example some kind of chunking operation? but importantly, we still need to be able to track the original source of the requests inferred and extracted.
```



## How to run

### First a subfolder, e.g., "test1" with the following structure:

```
test1/
    inputs/
    outputs/
```

### Process conversations

```
# For default
`python process_conversations.py`
# For more control, use the arguments for your project path and the number of exchanges to process
python process_conversations.py --project_path ./test1 --exchanges 50
```


