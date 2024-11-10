#include "NEAT/NN.h"
#include "pain.h"
#include <cmath>
#include <vector>

struct NodeInput {
  double outputValue = 0.0;
  bool activationFunctionApplied = false;
};

const std::vector<double> Genome::run(const std::vector<double> &inputs,
                                      int numInputs, int numOutputs) const
{
  std::unordered_map<int, NodeInput> nodeOutputs;
  nodeOutputs.reserve(m_links.size());

  // 1. Initialize input nodes with provided input values
  for (auto &node : m_neurons) {
    if (node.m_neuron_id < 0 && node.m_neuron_id >= -numInputs) {
      nodeOutputs[node.m_neuron_id] = {inputs[-(node.m_neuron_id + 1)],
                                       true}; // Input nodes are "activated"
    } else {
      nodeOutputs[node.m_neuron_id] = {
          node.m_bias, false}; // Initialize non-input nodes with bias
    }
  }

  // 2. Process each connection in topological order
  for (const auto &link : m_links) {
    if (!link.m_enable)
      continue; // Skip disabled links

    // Get or compute the output of the input node
    NodeInput &inputNode = nodeOutputs[link.m_InNodeId];
    if (!inputNode.activationFunctionApplied) {
      // Apply activation function if not yet applied
      const auto &nodeIt = std::find_if(
          m_neurons.begin(), m_neurons.end(),
          [&](const NodeGene &n) { return n.m_neuron_id == link.m_InNodeId; });

      // FIX: by commenting this, it means we can guarantee no dangling nodes
      if (nodeIt == m_neurons.end()) {
        PLOG_T("nodeIt->m_neuron_id = {}", nodeIt->m_neuron_id);
        PLOG_T("nodeIt->m_bias = {}", nodeIt->m_bias);
        ASSERT(nodeIt != m_neurons.end(),
               "NodeId = {} wasn't found inside m_neurons", link.m_InNodeId);
      }
      inputNode.outputValue =
          nodeIt->m_activationFunction.function(inputNode.outputValue);
      inputNode.activationFunctionApplied = true;
    }

    // Update the output of the target node by accumulating the weighted input
    nodeOutputs[link.m_OutNodeId].outputValue +=
        inputNode.outputValue * link.m_weight;
  }

  // 3. Apply activation to output nodes and collect outputs
  std::vector<double> outputs;
  for (int i = 0; i < numOutputs; ++i) {
    int outputNodeId = -(numInputs + i + 1);
    auto &outputNode = nodeOutputs[outputNodeId];
    if (!outputNode.activationFunctionApplied) {
      const auto &nodeIt = std::find_if(
          m_neurons.begin(), m_neurons.end(),
          [&](const NodeGene &n) { return n.m_neuron_id == outputNodeId; });
      ASSERT(nodeIt != m_neurons.end(),
             "outputNodeId = {} wasn't found inside m_neurons", outputNodeId);
      outputNode.outputValue =
          nodeIt->m_activationFunction(outputNode.outputValue);
      outputNode.activationFunctionApplied = true;
    }
    outputs.push_back(outputNode.outputValue);
  }
  return outputs;
}

int Genome::loadInnovation(int inputNodeId, int outputNodeId)
{
  int innov = -1;
  auto it = std::find_if(m_globalInnovations.begin(), m_globalInnovations.end(),
                         [&](const InnovationStatic &innov) {
                           return innov.m_InNodeId == inputNodeId &&
                                  innov.m_OutNodeId == outputNodeId;
                         });
  if (it != m_globalInnovations.end()) {
    // Innovation exists for the first connection
    innov = it->m_innovation;
  } else {
    // Create a new innovation for the first connection
    innov = m_globalInnovations.size();
    m_globalInnovations.emplace_back(innov, inputNodeId, outputNodeId);
  }
  ASSERT(innov != -1, "Couldn't get innovation but couldn't create either")
  return innov;
}

void Genome::addConnectionAndSort(ConnectionGene connGene, int numInputs)
{
  // check if there is a cycle
  if (hasPathDFS(connGene.m_OutNodeId, connGene.m_InNodeId)) {
    return;
  }

  m_links.push_back(std::move(connGene));

  topologySortNN(numInputs);
}

void Genome::topologySortNN(int numInputs)
{
  std::vector<int> sortedNodes;
  std::unordered_map<int, int> inDegree;

  // Count number of incomming connections
  for (const auto &link : m_links) {
    inDegree[link.m_OutNodeId]++;
  }

  // Start with inputs with inDegree of zero
  std::vector<int> zeroInDegreeNodes;
  zeroInDegreeNodes.reserve(numInputs);
  for (const auto &neuron : m_neurons) {
    if (inDegree[neuron.m_neuron_id] == 0) {
      zeroInDegreeNodes.push_back(neuron.m_neuron_id);
    }
  }

  // Process nodes in topological order
  while (!zeroInDegreeNodes.empty()) {
    int nodeId = zeroInDegreeNodes.back();
    zeroInDegreeNodes.pop_back();
    sortedNodes.push_back(nodeId);

    for (auto &link : m_links) {
      if (link.m_InNodeId == nodeId) {
        inDegree[link.m_OutNodeId]--;
        if (inDegree[link.m_OutNodeId] == 0) {
          zeroInDegreeNodes.push_back(link.m_OutNodeId);
        }
      }
    }
  }

  // Sort links based on sorted nodes order
  std::unordered_map<int, int> nodeOrder;
  for (size_t i = 0; i < sortedNodes.size(); ++i) {
    nodeOrder[sortedNodes[i]] = i;
  }
  std::sort(m_links.begin(), m_links.end(),
            [&](const ConnectionGene &a, const ConnectionGene &b) {
              return nodeOrder[a.m_InNodeId] < nodeOrder[b.m_InNodeId];
            });
}

bool Genome::hasPathDFS(int startNode, int targetNode) const
{
  // Use depth first search to check if there is a path from target to start
  std::unordered_set<int> visited;
  std::vector<int> stack = {targetNode};

  // just in case
  if (startNode == targetNode)
    return true;

  while (!stack.empty()) {
    int current = stack.back();
    stack.pop_back();

    if (current == startNode) {
      return true; // Somehow reached startNode, hence there is a path
    }

    visited.insert(current);

    for (const auto &link : m_links) {
      // Add all outgoing connections from the current node
      if (link.m_InNodeId == current && !visited.count(link.m_OutNodeId)) {
        stack.push_back(link.m_OutNodeId);
      }
    }
  }
  return false; // No path found, safe to add link
}

void Genome::addNode(ConnectionGene &oldConnGene, int numInputs, int numOutputs)
{
  // Disable the original connection
  oldConnGene.m_enable = false;

  // Create a new node ID
  int newNodeId = oldConnGene.m_innovation;

  // Create a new node with bias 0.0 and default sigmoid activation
  m_neurons.emplace_back(newNodeId, 0.0);

  // Search in m_globalInnovations for the first link
  int innovId1 = loadInnovation(oldConnGene.m_InNodeId, newNodeId);
  int innovId2 = loadInnovation(newNodeId, oldConnGene.m_OutNodeId);

  // Add two new connections:
  // 1. Connection from old input node to new node with weight 1.0
  ConnectionGene weight1 = {oldConnGene.m_InNodeId, newNodeId, 1.0, true,
                            innovId1};
  // 2. Connection from new node to the old output node with the original weight
  ConnectionGene weightOld = {newNodeId, oldConnGene.m_OutNodeId,
                              oldConnGene.m_weight, true, innovId2};

  auto oldConnGeneIt =
      std::find_if(m_links.begin(), m_links.end(),
                   [&oldConnGene](const ConnectionGene &conn) {
                     return conn.m_innovation == oldConnGene.m_innovation;
                   });

  // Insert both connections at once before the found position
  if (oldConnGeneIt != m_links.end()) {
    m_links.insert(oldConnGeneIt, {weight1, weightOld});
  }
  // HACK: this should not be necessary but idk
  topologySortNN(numInputs);
}

const bool Genome::willIsolateInNode(const ConnectionGene &link) const
{
  // Lambda function to check if a node has outgoing connections
  const auto hasOutputs = [&](int nodeId) {
    return std::any_of(
        m_links.begin(), m_links.end(),
        [&](const ConnectionGene &existingLink) {
          return existingLink.m_InNodeId == nodeId &&
                 existingLink.m_innovation !=
                     link.m_innovation; // prevents counting itself
        });
  };

  // Check if removing this link will isolate either the input or output node
  const bool willIsolateInNode = !hasOutputs(link.m_InNodeId);
  return willIsolateInNode;
}
const bool Genome::willIsolateOutNode(const ConnectionGene &link) const
{
  // Lambda function to check if a node has incoming connections
  const auto hasInputs = [&](int nodeId) {
    return std::any_of(
        m_links.begin(), m_links.end(),
        [&](const ConnectionGene &existingLink) {
          return existingLink.m_OutNodeId == nodeId &&
                 existingLink.m_innovation !=
                     link.m_innovation; // prevents counting itself
        });
  };

  // Check if removing this link will isolate either the input or output node
  const bool willIsolateOutNode = !hasInputs(link.m_OutNodeId);
  return willIsolateOutNode;
}

void Genome::removeConnection(ConnectionGene &link, int numInputs)
{
  if (willIsolateInNode(link) || willIsolateOutNode(link))
    return;
  // Find and remove the specified link
  auto linkIt = std::remove_if(
      m_links.begin(), m_links.end(), [&](const ConnectionGene &existingLink) {
        return existingLink.m_InNodeId == link.m_InNodeId &&
               existingLink.m_OutNodeId == link.m_OutNodeId &&
               existingLink.m_innovation == link.m_innovation;
      });
  m_links.erase(linkIt, m_links.end());
  topologySortNN(numInputs);
}

void Genome::removeNode(NodeGene &nodeGene, int numInputs)
{
  // Check if the node is an input or output
  if (nodeGene.m_neuron_id < 0)
    return;

  // check foregoing links of nodeGene
  bool isolationFound = std::any_of(
      m_links.begin(), m_links.end(), [&](const ConnectionGene &link) {
        return link.m_InNodeId == nodeGene.m_neuron_id &&
               willIsolateOutNode(link);
      });

  isolationFound =
      isolationFound ||
      std::any_of(m_links.begin(), m_links.end(),
                  [&](const ConnectionGene &link) {
                    return link.m_InNodeId == nodeGene.m_neuron_id &&
                           willIsolateInNode(link);
                  });

  if (isolationFound)
    return;

  // Remove the node from m_neurons
  auto neuronIt = std::remove_if(
      m_neurons.begin(), m_neurons.end(), [&](const NodeGene &node) {
        return node.m_neuron_id == nodeGene.m_neuron_id;
      });
  m_neurons.erase(neuronIt, m_neurons.end());

  // Remove all links associated with this node from m_links
  auto linksIt = std::remove_if(
      m_links.begin(), m_links.end(), [&](const ConnectionGene &link) {
        return link.m_InNodeId == nodeGene.m_neuron_id ||
               link.m_OutNodeId == nodeGene.m_neuron_id;
      });
  m_links.erase(linksIt, m_links.end());

  topologySortNN(numInputs);
}
