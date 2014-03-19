################################################################################
# The NeuralNetwork class represents a neural network. Its only job is to
# store and manage NNNode objects. NOTE TO SELF: Right now the neural network
# is only built to support a single layer.
################################################################################

package MachineLearning::NeuralNetwork;

use strict;
use warnings;
use Data::Dumper;
use MachineLearning::NNNode;

use base 'Exporter';
our @EXPORT = qw();

my $idCounter = 0;
my $biasNodeID = "bias";
my $debug = 0;

################################################################################
# Default constructor for a neural network object. The constructor does not 
# initialize the weights. But it does connect all of the nodes.
#   
#   @param an array of numbers of length n. 
#       a_0 is the number of input nodes.
#       a_1 to a_{n-2} is the number of nodes in each inner layer.
#       a_{n-1} is the number of output nodes.
#   @param the bias to set for all of the nodes.
#   @param the learning-rate to set for all of the nodes.
################################################################################
sub newNeuralNetwork {
    my $class = shift; 
    my $networkStructure = shift;
    my $bias = shift;
    my $learningRate = shift; # Also refered to as Nu. (the fancy greek 'n')

    my $self;
    $self->{USED_IDs} = ();    # A hash to store all of the used node ids
    
    my %aHash = (); $aHash{$biasNodeID} = undef; #Initialize hash for the registered nodes.
    $self->{REGISTERED_NODES}    = \%aHash;    # A hash of all the nodes that belong to the NN.

    #Needed so that we  can do the node connections later.
    $self->{NETWORK} = ();  #A list containing n-lists that contain the nodes.
                            #The list at index 0 is the input nodes.
                            #The lists at indexes 1 - (n - 2) are the layers.
                            #The list at index n-1 is the output nodes.
                            
    $self->{TARGET_OUTPUTS} = ();   #A hash of hashes that contains the target outputs of the 
                                    #output nodes depending on the value of the class.

    bless( $self, $class );
    
    #Construct the {NETWORK} list of lists
    my @network = map { nodeList($_, $learningRate, $self->{REGISTERED_NODES}) }
                        @$networkStructure;
    $self->{NETWORK} = \@network;
    
    $self->wireUpNeuralNetwork();   #Once the newtwork has been constructed wire 
                                    #up all of the nodes correctly.
    $self->setGlobalBias($bias);    #Attach the global bias node to the network.
    $self->setBiasForAllNodes();

    return $self;
}

################################################################################
# A "private" method to create a list of nodes.
#
#   @param the length of the list.
#   @param the learning rate of the nodes.
#   @param a hash reference to register the nodes to.
#
#   @param a list of all of the created nodes.
################################################################################
sub nodeList { 
    my $length = shift;
    my $learningRate = shift;
    my $registeredNodes = shift;
    my @list = ();
    
    for(my $i = 0; $i < $length; $i++) {
        my $node = MachineLearning::NNNode->newNNNode($idCounter, $learningRate);
        ${$registeredNodes}{$idCounter} = $node;
        push(@list, $node);
        $idCounter++;
    }
    
    return \@list;
}

################################################################################
# Get the output node IDs as a list.
################################################################################
sub getOutputNodeIDs {
    my $self = shift;
    
    my @ids = ();
    foreach(@${$self->{NETWORK}}[@{$self->{NETWORK}} - 1]) {
        my $outputNode = $_;
        
        push(@ids, $outputNode->getNodeID());
    }
    
    return @ids;
}

################################################################################
# Set the bias for all of the nodes. (pretty much creates a bogus node with
# an output set to one.
#
#	@param the bias to apply to all of the nodes.
################################################################################
sub setGlobalBias {
    my $self = shift;
    my $bias = shift;

    my $node = MachineLearning::NNNode->newNNNode($biasNodeID, undef);
    $node->setOutput($bias);
    $node->isBackPropagatable(0);

    ${ $self->{USED_IDs} }{$biasNodeID} = 1;
    ${ $self->{REGISTERED_NODES} }{$biasNodeID}    = $node;
}

################################################################################
# Set the input value for a dimension of all the input nodes.
#
#	@param The dimension of the value
#	@param The value of the dimension
################################################################################
sub setInputValueForInputDimension {
    my $self      = shift;
    my $dimension = shift;
    my $value     = shift;
    
    #Add or update the value inputs for the in nodes.
    foreach(@{${$self->{NETWORK}}[0]}) {
        my $inNode = $_;
        
        my $node = $inNode->getInEdge($dimension);
        
        #Set the output of the node if it exists
        #Otherwise add a new node.
        if(defined($node)) {
            $node->setOutput($value);
        }
        else {
            $node = MachineLearning::NNNode->newNNNode($dimension, undef);
            $node->isBackPropagatable(0);
            $node->setOutput($value);
            $inNode->addInEdge($node);
        }
    }
}

################################################################################
# Propagate forward through the network
################################################################################
sub propagateForward {
        my $self = shift;
                 
        print "\n>>>>>>>>>>>>Propagating Forward>>>>>>>>>>>>n" if($debug);
        
        foreach(@{${$self->{NETWORK}}[0]}) {
            $_->propagateForward();
        }
}

################################################################################
# Calculate the output errors after propagating forward through the network
################################################################################
sub calculateError {
    my $self = shift;
}

################################################################################
# Propagate backwards through the network
#
#   @param pass in the class of the classifiable that was used when the network
#       was propagated forward.
################################################################################
sub propagateBackwards {
        my $self = shift;
        my $classVal = shift;
        
        print "\n<<<<<<<<<<<<Propagating Backwards<<<<<<<<<<<<\n" if($debug);
        
        my %expectedVals = %{${$self->{TARGET_OUTPUTS}}{$classVal}};
        
        if((keys %expectedVals) > 0) {
            foreach my $key (keys %expectedVals) {
                my $outNode = ${$self->{REGISTERED_NODES}}{$key};
                
                $outNode->propagateBackwards($expectedVals{$key});
            }
        }
        else {
            
        }
}

################################################################################
# Predicts the class based on the output node values.
################################################################################
sub predictClass {
    my $self = shift;
    
    #Find the largest output value and round it up to 1. 
    my $count = @{$self->{NETWORK}};
    my $nodeID = undef; 
    my $nodeVal = undef;
    
    foreach(@{${$self->{NETWORK}}[$count - 1]}) {
        my $outputNode = $_;
        
        if(defined($nodeID)) {
                if($nodeVal < $outputNode->getOutput()) {
                    $nodeID = $outputNode->getNodeID();
                    $nodeVal = $outputNode->getOutput();
                }
        }
        else {
            $nodeID = $outputNode->getNodeID();
            $nodeVal = $outputNode->getOutput();
        }
    }
    
    #Now use the TARGET_OUTPUTS hash  to determin the class.
    $nodeVal = 1;
    my %targets = %{$self->{TARGET_OUTPUTS}};
    foreach my $classVal (keys %targets) {
        
        #If the expected value matches the node value 
        # then return the $classVal
        if(${$targets{$classVal}}{$nodeID} == $nodeVal) {
            return $classVal;
        }
    }
}

################################################################################
# Properly bind all the nodes in the network. Refresh all of the weights.
################################################################################
sub wireUpNeuralNetwork {
    my $self = shift;
    
    #Wire up all of the nodes.
    my $layerCount = @{$self->{NETWORK}};
    for(my $i = 0; $i < $layerCount - 1; $i++) {
        foreach(@{${$self->{NETWORK}}[$i]}) {
            my $currLayerNode = $_;
            
            foreach(@{${$self->{NETWORK}}[$i+1]}) {
                my $nextLayerNode = $_;
                
                $nextLayerNode->addInEdge($currLayerNode);
                $currLayerNode->addOutEdge($nextLayerNode);                
            }
        }
    }
}

################################################################################
# Wires all of the bias nodes to each node in the network.
################################################################################
sub setBiasForAllNodes {
    my $self = shift;
    
    my $biasNode = ${$self->{REGISTERED_NODES}}{$biasNodeID};
    foreach my $k (keys %{$self->{REGISTERED_NODES}}) {
        my $node = ${$self->{REGISTERED_NODES}}{$k};
        $node->addInEdge($biasNode);
    }
}

################################################################################
# Set the learning rate for all of the nodes in the network. (learning rate == 
# nu)
#
#       @param the learning rate (nu)
################################################################################
sub setNuForAllNodes {
    my $self = shift;
    my $nu = shift;
    
    foreach my $k (keys %{$self->{REGISTERED_NODES}}) {
        my $node = ${$self->{REGISTERED_NODES}}{$k};
        $node->Nu($nu);
    }    
}

################################################################################
# Set the expected output values for the output nodes.
#
#   @param an array reference of the possible output values.
################################################################################
sub setExpectedOutputValues {
    my $self = shift;
    my $possibleValues = shift;
    
    my %targetOutputs = (); #The expected output node combinations depending on the value of the class.
    for(my $i = 0; $i < @{$possibleValues}; $i++) {
        my $value = $$possibleValues[$i];
        
        my %nodeOutputs = (); #the values the output nodes need to have depending on the target output.
        for(my $j = 0; $j < @{${$self->{NETWORK}}[@{$self->{NETWORK}} - 1]}; $j++) {
            my $outNode = ${${$self->{NETWORK}}[@{$self->{NETWORK}} - 1]}[$j];
            
            if($j == $i) {
                $nodeOutputs{$outNode->getNodeID()} = 1;
            }
            else {
                $nodeOutputs{$outNode->getNodeID()} = 0;
            }
        }
        
        $targetOutputs{$value} = \%nodeOutputs;
    }
    
    $self->{TARGET_OUTPUTS} = \%targetOutputs; #Set the target outputs.
    
    # print Dumper \%targetOutputs;
}

################################################################################
# Goes through all of the nodes and initializes the node weights
# 
#   @param the bounds for the randomnization of the weights.
################################################################################
sub initializeWeights {
    my $self = shift;
    my $bound = shift;
    
    foreach my $k (keys %{$self->{REGISTERED_NODES}}) {
        my $node = ${$self->{REGISTERED_NODES}}{$k};
        $node->generateInWeights($bound);
    }
}

################################################################################
# Prints out the network.
################################################################################
sub toString {
    my $self = shift;
    my $layerCount = @{$self->{NETWORK}};
    
    print "\n\n################################################################################\n";
    print "Counter: $idCounter\n";

    my @t     = ( keys %{ $self->{REGISTERED_NODES} } );
    my $count = @t;
    print "Node Count: $count\n";

    print "\n\n>>>>Input Nodes<<<<\n";
    foreach(@{${$self->{NETWORK}}[0]}) {
        my $node = $_;
        $node->toString();
    }

    print "\n\n>>>>Inner Nodes<<<<\n";
    for(my $i = 1; $i < $layerCount - 1; $i++) {
        print "\n\n****LAYER  $i****\n";
        foreach(@{${$self->{NETWORK}}[$i]}) {
            my $node = $_;
            $node->toString();
        }
    }

    print "\n\n>>>>Output Nodes<<<<\n";
    foreach(@{${$self->{NETWORK}}[$layerCount - 1]}) {
        my $node = $_;
        $node->toString();
    }
    print "\n################################################################################\n\n";
}

return 1;


