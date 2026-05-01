import pandas as pd

# Increase pandas display area
pd.set_option('display.max_rows', 500)  # Show up to 500 rows
pd.set_option('display.max_columns', 100)  # Show up to 100 columns
pd.set_option('display.width', 1000)  # Width of display in characters
pd.set_option('display.max_colwidth', 200)  # Max width for column content
pd.set_option('display.max_info_columns', 100)  # Columns to show in info()
pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple lines

print("✓ Pandas display options configured for larger output")


def extract_agent_interactions(response):
    """
    Extract agent interactions and handoffs from a LangGraph response.

    Args:
        response: Dictionary containing 'messages' list from LangGraph execution

    Returns:
        List of interaction dictionaries with details about each agent handoff
    """
    interactions = []
    messages = response.get('messages', [])

    for i, message in enumerate(messages):
        interaction = {}

        # Get message type
        message_type = type(message).__name__

        # Extract agent name if available
        agent_name = getattr(message, 'name', None)

        # Check for tool calls (transfers/handoffs)
        tool_calls = getattr(message, 'tool_calls', None)

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get('name', '')

                # Check if it's a transfer/handoff
                if 'transfer' in tool_name.lower() or 'handoff' in tool_name.lower():
                    interaction = {
                        'type': 'handoff',
                        'from_agent': agent_name,
                        'to_agent': tool_name.replace('transfer_to_', '').replace('transfer_back_to_', ''),
                        'tool_call': tool_name,
                        'tool_call_id': tool_call.get('id', ''),
                        'message_index': i,
                        'content': getattr(message, 'content', ''),
                        'message_type': message_type
                    }
                    interactions.append(interaction)

        # Check for ToolMessage confirming transfers
        if message_type == 'ToolMessage':
            content = getattr(message, 'content', '')
            tool_name = getattr(message, 'name', '')

            if 'transfer' in content.lower() or 'transfer' in tool_name.lower():
                interaction = {
                    'type': 'transfer_confirmation',
                    'tool_name': tool_name,
                    'content': content,
                    'message_index': i,
                    'tool_call_id': getattr(message, 'tool_call_id', ''),
                    'message_type': message_type
                }
                interactions.append(interaction)

        # Track agent messages
        if message_type == 'AIMessage' and agent_name:
            # Check if this is a handoff back (some systems use response_metadata)
            response_metadata = getattr(message, 'response_metadata', {})
            if response_metadata.get('__is_handoff_back'):
                interaction = {
                    'type': 'handoff_back',
                    'from_agent': agent_name,
                    'message_index': i,
                    'content': getattr(message, 'content', ''),
                    'message_type': message_type
                }
                interactions.append(interaction)

    return interactions


def print_agent_interactions(interactions):
    """
    Pretty print agent interactions.
    """
    print("=" * 80)
    print("AGENT INTERACTIONS & HANDOFFS")
    print("=" * 80)

    for idx, interaction in enumerate(interactions, 1):
        print(f"\n[{idx}] {interaction['type'].upper()}")
        print("-" * 80)

        for key, value in interaction.items():
            if key != 'type':
                print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print(f"Total interactions: {len(interactions)}")
    print("=" * 80)


def visualize_agent_flow(response):
    """
    Create a simple flow visualization of agent handoffs with token usage tracking.
    """
    messages = response.get('messages', [])
    flow = []
    current_agent = None

    # Cumulative token tracking
    cumulative_input_tokens = 0
    cumulative_output_tokens = 0
    cumulative_total_tokens = 0

    # Interaction tracking
    total_agent_actions = 0
    total_transfers = 0

    for message in messages:
        message_type = type(message).__name__
        agent_name = getattr(message, 'name', None)
        tool_calls = getattr(message, 'tool_calls', None)

        # Extract token usage if available
        usage_metadata = getattr(message, 'usage_metadata', None)
        step_tokens = None

        if usage_metadata:
            input_tokens = usage_metadata.get('input_tokens', 0)
            output_tokens = usage_metadata.get('output_tokens', 0)
            total_tokens = usage_metadata.get('total_tokens', 0)

            cumulative_input_tokens += input_tokens
            cumulative_output_tokens += output_tokens
            cumulative_total_tokens += total_tokens

            step_tokens = {
                'input': input_tokens,
                'output': output_tokens,
                'total': total_tokens
            }

        # Track current agent - show ALL agent messages, not just when agent changes
        if message_type == 'AIMessage' and agent_name:
            # Update current agent tracking
            if agent_name != current_agent:
                current_agent = agent_name

            # Always add agent interaction to flow (even if same agent)
            total_agent_actions += 1
            content_preview = getattr(message, 'content', '')[:80]
            if content_preview:
                content_preview = content_preview.replace('\n', ' ') + '...'
            flow.append(f"[{agent_name}]")
            if content_preview:
                flow.append(f"  └─ Says: {content_preview}")

            # Add token usage for this step
            if step_tokens:
                flow.append(f"  └─ Tokens: Input={step_tokens['input']}, Output={step_tokens['output']}, Total={step_tokens['total']}")
                flow.append(f"  └─ Cumulative: Input={cumulative_input_tokens}, Output={cumulative_output_tokens}, Total={cumulative_total_tokens}")

        # Track transfers
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.get('name', '')
                if 'transfer' in tool_name.lower():
                    target = tool_name.replace('transfer_to_', '').replace('transfer_back_to_', '')
                    total_transfers += 1
                    flow.append(f"  └─ Transfers to: {target}")

    print("\n" + "=" * 120)
    print("AGENT FLOW VISUALIZATION")
    print("=" * 120)
    for step in flow:
        print(step)
    print("=" * 120)
    print(f"\nTotal interactions: {total_agent_actions + total_transfers} (Agent actions: {total_agent_actions}, Transfers: {total_transfers})")
    print(f"FINAL CUMULATIVE TOKENS: Input={cumulative_input_tokens}, Output={cumulative_output_tokens}, Total={cumulative_total_tokens}")
    print("=" * 120 + "\n")