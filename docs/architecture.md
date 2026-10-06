# Ron-0 Architecture

## Goal
Build Ron as an agent runtime rather than a provider-specific chatbot.

## Layers
1. Conversation Engine — owns turns and creates runtime requests.
2. Ron Core — coordinates understanding, planning, execution, verification, and response.
3. Memory — short-term and long-term memory behind a replaceable interface.
4. Retrieval — retrieves relevant knowledge without flooding model context.
5. Tools — explicit capability contracts with inputs, outputs, and errors.
6. Model Router — selects a configured provider through a common interface.

## Initial contracts
- ModelProvider
- MemoryStore
- Retriever
- Tool
- Planner
- AgentRuntime

## Execution lifecycle
1. Receive request.
2. Build context.
3. Decide whether planning, retrieval, or tools are needed.
4. Execute bounded actions.
5. Verify results.
6. Produce response.
7. Record useful memory.

## Design rule
No external service becomes the architectural center of Ron. External services are adapters around the GitHub-owned codebase.
