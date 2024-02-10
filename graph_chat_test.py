import random  # noqa E402

import matplotlib.pyplot as plt  # noqa E402
import networkx as nx  # noqa E402

import autogen  # noqa E402
from autogen.agentchat.assistant_agent import AssistantAgent  # noqa E402
from autogen.agentchat.groupchat import GroupChat, Agent  # noqa E402
from autogen.graph_utils import visualize_speaker_transitions_dict  # noqa E402

print(autogen.__version__)

# The default config list in notebook.
config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-turbo-preview"],
    },
)


# llm config
llm_config = {"config_list": config_list_gpt4, "cache_seed": 100}

# Create an empty directed graph
agents = []
speaker_transitions_dict = {}
secret_values = {}


agents.append(
    AssistantAgent( name=  'A1',
                    system_message= "You are a helpful agent, you are always succinct and direct to the point. You dislike riddles.",
                    llm_config=llm_config
                        )
) 
speaker_transitions_dict[agents[-1]] = []

agents.append(
    AssistantAgent( name=  'A2',
                    system_message= "You are an unhelpful agent, you always disagree with A1",
                    llm_config=llm_config
                        ) 
)
speaker_transitions_dict[agents[-1]] = []

agents.append(
    AssistantAgent( name=  'A3',
                   system_message= "You are a very wise agent, however you always speak in riddles.",
                   llm_config=llm_config
                    ) 
)
speaker_transitions_dict[agents[-1]] = []

agents.append(
    AssistantAgent( name=  'B1',
                   system_message= "You are a helpful agent you are always succinct and direct to the point. You dislike riddles.",
                   llm_config=llm_config
                    ) 
)
speaker_transitions_dict[agents[-1]] = []

agents.append(
    AssistantAgent( name=  'B2',
                   system_message= "You are an unhelpful agent, you always disagree with A1",
                   llm_config=llm_config
                    ) 
)
speaker_transitions_dict[agents[-1]] = []

agents.append(
    AssistantAgent( name=  'B3',
                   system_message= "You are a very wise agent, however you always speak in riddles.",
                   llm_config=llm_config
                    ) 
)
speaker_transitions_dict[agents[-1]] = []


agents.append(
    AssistantAgent( name=  'C1',
                   system_message= "You are a helpful agent you are always succinct and direct to the point. You dislike riddles.",
                   llm_config=llm_config
                    ) 
)
speaker_transitions_dict[agents[-1]] = []

agents.append(
    AssistantAgent( name=  'C2',
                   system_message= "You are an unhelpful agent, you always disagree with A1",
                   llm_config=llm_config
                    ) 
)
speaker_transitions_dict[agents[-1]] = []

agents.append(
    AssistantAgent( name=  'C3',
                   system_message= "You are a very wise agent, however you always speak in riddles.",
                   llm_config=llm_config
                    ) 
)
speaker_transitions_dict[agents[-1]] = []



def get_agent_of_name(agents, name) -> Agent:
    for agent in agents:
        if agent.name == name:
            return agent


speaker_transitions_dict[get_agent_of_name(agents, "A1")].append(get_agent_of_name(agents, name="A2"))
speaker_transitions_dict[get_agent_of_name(agents, "A2")].append(get_agent_of_name(agents, name="A3"))
speaker_transitions_dict[get_agent_of_name(agents, "A3")].append(get_agent_of_name(agents, name="A1"))


speaker_transitions_dict[get_agent_of_name(agents, "B1")].append(get_agent_of_name(agents, name="B2"))
speaker_transitions_dict[get_agent_of_name(agents, "B2")].append(get_agent_of_name(agents, name="B3"))
speaker_transitions_dict[get_agent_of_name(agents, "B3")].append(get_agent_of_name(agents, name="B1"))


speaker_transitions_dict[get_agent_of_name(agents, "C1")].append(get_agent_of_name(agents, name="C2"))
speaker_transitions_dict[get_agent_of_name(agents, "C2")].append(get_agent_of_name(agents, name="C3"))
speaker_transitions_dict[get_agent_of_name(agents, "C3")].append(get_agent_of_name(agents, name="C1"))

# Adding edges between teams
speaker_transitions_dict[get_agent_of_name(agents, "A1")].append(get_agent_of_name(agents, name="B1"))
speaker_transitions_dict[get_agent_of_name(agents, "A1")].append(get_agent_of_name(agents, name="C1"))
speaker_transitions_dict[get_agent_of_name(agents, "B1")].append(get_agent_of_name(agents, name="A1"))
speaker_transitions_dict[get_agent_of_name(agents, "B1")].append(get_agent_of_name(agents, name="C1"))
speaker_transitions_dict[get_agent_of_name(agents, "C1")].append(get_agent_of_name(agents, name="A1"))
speaker_transitions_dict[get_agent_of_name(agents, "C1")].append(get_agent_of_name(agents, name="B1"))


# Visualization only

graph = nx.DiGraph()

# Add nodes
graph.add_nodes_from([agent.name for agent in agents])

# Add edges
for key, value in speaker_transitions_dict.items():
    for agent in value:
        graph.add_edge(key.name, agent.name)

# Visualize


# Draw the graph with secret values annotated
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(graph)  # positions for all nodes

# Draw nodes with their colors
nx.draw(graph, pos, with_labels=True, font_weight="bold")

# Annotate secret values
#for node, (x, y) in pos.items():
#    secret_value = secret_values[node]
#    plt.text(x, y + 0.1, s=f"Secret: {secret_value}", horizontalalignment="center")

plt.show()


# Termination message detection


def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False


# Terminates the conversation when TERMINATE is detected.
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="Terminator admin.",
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
    human_input_mode="NEVER",
)

agents.append(user_proxy)

group_chat = GroupChat(
    agents=agents,
    messages=[],
    max_round=20,
    allowed_or_disallowed_speaker_transitions=speaker_transitions_dict,
    speaker_transitions_type="allowed",
)


# Create the manager
manager = autogen.GroupChatManager(
    groupchat=group_chat, llm_config=llm_config, code_execution_config=False, is_termination_msg=is_termination_msg
)


# Initiates the chat with Alice
agents[0].initiate_chat(
    manager,
    message="""
                        There are 3 teams of 3 Agents. Each team must decide on their favorite farm animal. I will now start with my team.
                        NEXT: A1""",
)