#include "NEAT/NN.h"

struct NodeInput {
  double outputValue = 0.0;
  bool activationFunctionApplied = false;
};

std::vector<double> Genome::run(const std::vector<double> &inputs)
{

  std::unordered_map<int, NodeInput> nodeOutputs;

  // 1. Initialize input nodes with provided input values
  int inputIndex = 0;
  for (auto &node : m_neurons) {
    if (inputIndex < m_num_inputs) {
      nodeOutputs[node.m_neuron_id] = {inputs[inputIndex++],
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
      if (nodeIt != m_neurons.end()) {
        inputNode.outputValue =
            nodeIt->m_activationFunction(inputNode.outputValue);
      }
      inputNode.activationFunctionApplied = true;
    }

    // Update the output of the target node by accumulating the weighted input
    nodeOutputs[link.m_OutNodeId].outputValue +=
        inputNode.outputValue * link.m_weight;
  }

  // 3. Apply activation to output nodes and collect outputs
  std::vector<double> outputs;
  for (int i = 0; i < m_num_ouputs; ++i) {
    int outputNodeId = m_num_inputs + i;
    auto &outputNode = nodeOutputs[outputNodeId];
    if (!outputNode.activationFunctionApplied) {
      const auto &node = std::find_if(
          m_neurons.begin(), m_neurons.end(),
          [&](const NodeGene &n) { return n.m_neuron_id == outputNodeId; });
      if (node != m_neurons.end()) {
        outputNode.outputValue =
            node->m_activationFunction(outputNode.outputValue);
      }
    }
    outputs.push_back(outputNode.outputValue);
  }
  return outputs;
}

// WARNING: This will create a new layer if necessary
int Genome::needLowerNode(int nodeId)
{
  const int inLayerId = getNodeLayer(nodeId);
  // getting the layerId from the layer below
  auto layerPos = std::find(m_layers.begin(), m_layers.end(), inLayerId);
  const int lowerLayer = *(--layerPos);

  const bool hasLowerLayerConnection = std::any_of(
      m_links.begin(), m_links.end(), [&](const ConnectionGene &link) {
        return link.m_OutNodeId == nodeId && lowerLayer == inLayerId;
      });

  if (hasLowerLayerConnection) {
    // Create a new layer if m_InNodeId is already connected to the layer
    // below
    int newLayerId = m_layers.size();
    m_layers.insert(layerPos, newLayerId);
    return newLayerId;
  } else {
    // Move m_InNodeId to a lower layer
    auto lowerLayerIt =
        std::lower_bound(m_layers.begin(), m_layers.end(), inLayerId);
    ASSERT(lowerLayerIt != m_layers.begin(),
           "LayerId can not be lower than that");
    return *(--lowerLayerIt);
  }
}

void Genome::addConnection(ConnectionGene connGene)
{
  // Find the input and output nodes by their IDs

  int &inLayerId = getNodeLayer(connGene.m_InNodeId);
  int &outLayerId = getNodeLayer(connGene.m_OutNodeId);

  // Calculate the index distance between the input and output layers
  int indexDistance = getLayerIndexDistance(inLayerId, outLayerId);

  // ------------------------------------------------------------------------ //
  // Case 1: We don't know if the mutated link is in the right direction
  if (indexDistance < 0) {
    std::swap(inLayerId, outLayerId);
    indexDistance = -indexDistance; // no need to recalculate
    // FIX: THIS IS TEMPORARY! DELETE THIS AFTER YOU CHECK
    ASSERT(getLayerIndexDistance(getNodeLayer(connGene.m_InNodeId),
                                 getNodeLayer(connGene.m_OutNodeId)) < 0,
           "If you are seeing this it means you need to make two swaps");
  }
  // ------------------------------------------------------------------------ //
  if (indexDistance > 0) {
    // Case 2: Check if the connection goes to a higher layer
    // Find the position to insert in m_links based on layer ordering
    auto pos = std::lower_bound(
        m_links.begin(), m_links.end(), connGene,
        [&](const ConnectionGene &a, const ConnectionGene &b) {
          return getLayerIndexDistance(a.m_InNodeId, b.m_InNodeId) > 0;
        });

    // Insert the new connection at the found position
    m_links.insert(pos, std::move(connGene));

    // ----------------------------------------------------------------------
  } else if (indexDistance == 0) {
    // Case 3: If inLayerId equals outLayerId
    inLayerId = needLowerNode(connGene.m_InNodeId);

    // Re-insert the connection now that m_InNodeId's layer is adjusted
    auto pos = std::lower_bound(
        m_links.begin(), m_links.end(), connGene,
        [&](const ConnectionGene &a, const ConnectionGene &b) {
          return getLayerIndexDistance(a.m_InNodeId, b.m_InNodeId) > 0;
        });

    m_links.insert(pos, std::move(connGene));
  }
}

void Genome::addNode(ConnectionGene &oldGene)
{
  // Disable the original connection
  oldGene.m_enable = false;

  // Create a new node ID
  int newNodeId = oldGene.m_innovation;

  // Will create a new layer or use an existing layer below
  int newNodeLayer = needLowerNode(oldGene.m_InNodeId);
  // calculate layer position

  // Create a new node with bias 0.0 and default sigmoid activation
  m_neurons.emplace_back(newNodeId, 0.0, newNodeLayer);

  int innovId1 = -1;
  int innovId2 = -1;

  // Search in m_globalInnovations for the first link
  auto it1 =
      std::find_if(m_globalInnovations.begin(), m_globalInnovations.end(),
                   [&](const InnovationStatic &innov) {
                     return innov.m_InNodeId == oldGene.m_InNodeId &&
                            innov.m_OutNodeId == newNodeId;
                   });
  if (it1 != m_globalInnovations.end()) {
    // Innovation exists for the first connection
    innovId1 = it1->m_innovation;
  } else {
    // Create a new innovation for the first connection
    innovId1 = m_globalInnovations.size();
    m_globalInnovations.emplace_back(innovId1, oldGene.m_InNodeId, newNodeId);
  }

  // Search in m_globalInnovations for the second link
  auto it2 =
      std::find_if(m_globalInnovations.begin(), m_globalInnovations.end(),
                   [&](const InnovationStatic &innov) {
                     return innov.m_InNodeId == newNodeId &&
                            innov.m_OutNodeId == oldGene.m_OutNodeId;
                   });
  if (it2 != m_globalInnovations.end()) {
    // Innovation exists for the second connection
    innovId2 = it2->m_innovation;
  } else {
    // Create a new innovation for the second connection
    innovId2 = m_globalInnovations.size();
    m_globalInnovations.emplace_back(innovId2, newNodeId, oldGene.m_OutNodeId);
  }

  // Add two new connections:
  // 1. Connection from old input node to new node with weight 1.0
  m_links.emplace_back(oldGene.m_InNodeId, newNodeId, 1.0, true, innovId1);

  // 2. Connection from new node to the old output node with the original weight
  m_links.emplace_back(newNodeId, oldGene.m_OutNodeId, oldGene.m_weight, true,
                       innovId2);
}

void Genome::removeNode(NodeGene &nodeGene)
{
  // Check if the node is an input or output
  auto nodeGeneIndex = std::find_if(
      m_neurons.begin(), m_neurons.end(), [&](const NodeGene &node) {
        return node.m_neuron_id == nodeGene.m_neuron_id;
      });

  int nodeIndex = std::distance(m_neurons.begin(), nodeGeneIndex);
  // check if is input or output node. Will do nothing
  if (nodeIndex < m_num_inputs ||
      nodeIndex >= (m_neurons.size() - m_num_ouputs)) {
    return;
  }

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
}

void Genome::removeConnection(ConnectionGene &link)
{
  // Find and remove the specified link
  auto linkIt = std::remove_if(
      m_links.begin(), m_links.end(), [&](const ConnectionGene &existingLink) {
        return existingLink.m_InNodeId == link.m_InNodeId &&
               existingLink.m_OutNodeId == link.m_OutNodeId &&
               existingLink.m_innovation == link.m_innovation;
      });
  m_links.erase(linkIt, m_links.end());
}
