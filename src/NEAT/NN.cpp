#include "NEAT/NN.h"
#include "pain.h"
#include <iterator>
#include <vector>

struct NodeInput {
  double outputValue = 0.0;
  bool activationFunctionApplied = false;
};

std::vector<double> Genome::run(const std::vector<double> &inputs,
                                int numInputs, int numOutputs)
{
  std::map<int, NodeInput> nodeOutputs;
  // nodeOutputs.reserve(m_links.size());

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

      ASSERT(nodeIt != m_neurons.end(),
             "outputNodeId = {} wasn't found inside m_neurons",
             link.m_InNodeId);
      inputNode.outputValue =
          nodeIt->m_activationFunction(inputNode.outputValue);
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

// WARNING: This will create a new layer if necessary
int Genome::sameLayerLinkAdition(int nodeId)
{
  const int layerId = getNodeLayer(nodeId);
  auto layerPos = std::find(m_layers.begin(), m_layers.end(), layerId);

  // getting the layerId from the layer below
  const int lowerLayerId = *(--layerPos);

  // checks if anyone already exists
  const bool nodeHasLowerLayerConnection = std::any_of(
      m_links.begin(), m_links.end(), [&](const ConnectionGene &link) {
        return link.m_OutNodeId == nodeId &&
               getNodeLayer(link.m_InNodeId) == lowerLayerId;
      });

  if (nodeHasLowerLayerConnection) {
    // Create a new layer if m_InNodeId is already connected to the layer
    // below
    int newLayerId = m_layers.size();
    m_layers.insert(layerPos, newLayerId);
    return newLayerId;
  } else {
    // Move m_InNodeId to a lower layer
    return lowerLayerId;
  }
}

// WARNING: This will create a new layer if necessary
int Genome::lowerLayerNodeAdition(int inputNodeId, int outputNodeId)
{
  const int inputLayerId = getNodeLayer(inputNodeId);
  const int outputLayerId = getNodeLayer(outputNodeId);

  auto inputLayerIndex =
      std::find(m_layers.begin(), m_layers.end(), inputLayerId);
  const auto outputLayerIndex =
      std::find(m_layers.begin(), m_layers.end(), outputLayerId);

  int distance = std::distance(inputLayerIndex, outputLayerIndex);
  // this means there is at least one more node betwen input and output layer
  if (distance > 1) {
    return *(++inputLayerIndex);
  } else if (distance == 1) { // means there is no layer betwen both layers
    int newLayerId = m_layers.size();
    m_layers.insert(outputLayerIndex, newLayerId);
    return newLayerId;
  } else {
    PLOG_E("Distance between added nodes during node addition is less than 1");
    throw("Distance between added nodes during node addition is less than 1");
  }
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
    std::swap(connGene.m_InNodeId, connGene.m_OutNodeId);
    indexDistance = -indexDistance; // no need to recalculate
  }
  // ------------------------------------------------------------------------ //
  if (indexDistance > 0) {
    // Case 2: Check if the connection goes to a higher layer
    // Find the position to insert in m_links based on layer ordering
    auto pos = std::lower_bound(
        m_links.begin(), m_links.end(), connGene,
        [&](const ConnectionGene &a, const ConnectionGene &input) {
          return getLayerIndexDistance(getNodeLayer(a.m_InNodeId),
                                       getNodeLayer(input.m_InNodeId)) > 0;
        });

    // Insert the new connection at the found position
    m_links.insert(pos, std::move(connGene));

    // ----------------------------------------------------------------------
  } else if (indexDistance == 0) {
    // Case 3: If inLayerId equals outLayerId
    inLayerId = sameLayerLinkAdition(connGene.m_InNodeId);

    // Re-insert the connection now that m_InNodeId's layer is adjusted
    auto pos = std::lower_bound(
        m_links.begin(), m_links.end(), connGene,
        [&](const ConnectionGene &a, const ConnectionGene &input) {
          return getLayerIndexDistance(getNodeLayer(a.m_InNodeId),
                                       getNodeLayer(input.m_InNodeId)) > 0;
        });

    m_links.insert(pos, std::move(connGene));
  }
}

void Genome::addNode(ConnectionGene &oldConnGene, int numInputs, int numOutputs)
{
  // Disable the original connection
  oldConnGene.m_enable = false;

  // Create a new node ID
  int newNodeId = oldConnGene.m_innovation;

  // Will create a new layer or use an existing layer below
  int newNodeLayer =
      lowerLayerNodeAdition(oldConnGene.m_InNodeId, oldConnGene.m_OutNodeId);
  // calculate layer position

  // Create a new node with bias 0.0 and default sigmoid activation
  m_neurons.emplace_back(newNodeId, 0.0, newNodeLayer);

  // Search in m_globalInnovations for the first link
  int innovId1 = loadInnovation(oldConnGene.m_InNodeId, newNodeId);
  int innovId2 = loadInnovation(newNodeId, oldConnGene.m_OutNodeId);

  // Add two new connections:
  // 1. Connection from old input node to new node with weight 1.0
  m_links.emplace_back(oldConnGene.m_InNodeId, newNodeId, 1.0, true, innovId1);

  // 2. Connection from new node to the old output node with the original weight
  m_links.emplace_back(newNodeId, oldConnGene.m_OutNodeId, oldConnGene.m_weight,
                       true, innovId2);
}

const bool willIsolateNodesIfRemoved(const std::vector<ConnectionGene> &m_links,
                                     ConnectionGene &link)
{
  // Lambda function to check if a node has outgoing connections
  const auto hasOutputs = [&](int nodeId) {
    return std::any_of(m_links.begin(), m_links.end(),
                       [&](const ConnectionGene &existingLink) {
                         return existingLink.m_InNodeId == nodeId &&
                                existingLink.m_innovation != link.m_innovation;
                       });
  };

  // Lambda function to check if a node has incoming connections
  const auto hasInputs = [&](int nodeId) {
    return std::any_of(m_links.begin(), m_links.end(),
                       [&](const ConnectionGene &existingLink) {
                         return existingLink.m_OutNodeId == nodeId &&
                                existingLink.m_innovation != link.m_innovation;
                       });
  };

  // Check if removing this link will isolate either the input or output node
  const bool willIsolateInNode = !hasOutputs(link.m_InNodeId);
  const bool willIsolateOutNode = !hasInputs(link.m_OutNodeId);
  return willIsolateOutNode && willIsolateInNode;
}

void Genome::removeConnection(ConnectionGene &link)
{
  if (willIsolateNodesIfRemoved(m_links, link))
    return;
  // Find and remove the specified link
  auto linkIt = std::remove_if(
      m_links.begin(), m_links.end(), [&](const ConnectionGene &existingLink) {
        return existingLink.m_InNodeId == link.m_InNodeId &&
               existingLink.m_OutNodeId == link.m_OutNodeId &&
               existingLink.m_innovation == link.m_innovation;
      });
  m_links.erase(linkIt, m_links.end());
}

void Genome::removeNode(NodeGene &nodeGene)
{
  // Check if the node is an input or output
  if (nodeGene.m_neuron_id < 0)
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
}
