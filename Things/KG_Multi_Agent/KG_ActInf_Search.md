Thoughtfully design Pythonic pseudo-code Active Inference POMDP agent augmented with an LLM in an agentic workflow in an action-perception loop with the following specifications:
- A prompt is sent to an LLM. The LLM prompting includes a system prompt denoting which key terms to return in its response to denote the inferred hidden state to which the prompt the LLM received relates. The key terms returned are then the structured discrete observations the POMDP agent receives.
- the POMDP agent infer a single hidden state factor where each discrete hidden state represents a node in an obsidian [[linked]] knowledge graph.
- Number of actions for the single hidden state factor are equivalent to number of hidden states.
- Policy inference over a scalar temporal horizon of policy_horizon means that the resulting selected policy is a sequential order of [[links]] in the knowledge graph to visit and collect information.
- The markdown contents of the nodes implicated in the agent's selected policy are then sequentially combined in accordance with the policy and used as graph RAG material for the LLM to answer the original query.

# Pythonic pseudo-code for an Active Inference POMDP agent with LLM augmentation

from pymdp import Agent  # POMDP active inference library
import obsidian_api  # Hypothetical API for interacting with Obsidian knowledge graphs
from llm_api import LLM  # Hypothetical API for interacting with the LLM

# Initialize the POMDP agent
hidden_states = ["Node1", "Node2", "Node3"]  # Example nodes in the Obsidian knowledge graph
actions = hidden_states  # Each action corresponds to visiting a node
policy_horizon = 3  # Temporal horizon for policy inference

# Define the generative model components
A = ...  # Observation likelihood matrix (to be defined based on key terms)
B = ...  # Transition matrix (e.g., uniform transitions between nodes)
C = ...  # Reward model (e.g., prioritize nodes with high information gain)
D = ...  # Prior over hidden states (e.g., uniform or based on prior knowledge)

agent = Agent(A=A, B=B, C=C, D=D)

# Define the LLM and its system prompt
llm_system_prompt = """
You are an assistant helping to infer hidden states in a knowledge graph. 
Return key terms that map observations to specific nodes in the graph.
"""

llm = LLM(system_prompt=llm_system_prompt)

# Action-perception loop
def action_perception_loop(query):
    """
    Main workflow for the Active Inference agent augmented with an LLM.
    """
    # Step 1: Prompt the LLM with the query
    llm_response = llm.prompt(query)
    structured_observation = extract_key_terms(llm_response)  # Extract discrete observations
    
    # Step 2: Update agent's belief state based on observation
    agent.update_beliefs(observation=structured_observation)
    
    # Step 3: Infer policy over a scalar temporal horizon
    policy = agent.infer_policy(horizon=policy_horizon)
    
    # Step 4: Execute the policy by visiting nodes in the knowledge graph
    combined_markdown_content = ""
    for action in policy:
        markdown_content = obsidian_api.get_node_content(action)  # Fetch node content
        combined_markdown_content += markdown_content + "\n"
    
    # Step 5: Use combined markdown content as RAG material for LLM to answer the query
    final_answer = llm.prompt(query, context=combined_markdown_content)
    
    return final_answer

# Helper function to extract key terms from LLM response
def extract_key_terms(response):
    """
    Parse LLM response to extract structured discrete observations.
    """
    key_terms = response.get("key_terms", [])
    return key_terms

# Example usage of the system
query = "Explain the relationship between quantum mechanics and general relativity."
answer = action_perception_loop(query)
print(answer)
