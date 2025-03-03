# MKG_Multi_Agent

This is a multi-agent system for a knowledge graph.



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


