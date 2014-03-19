################################################################################
# NNNode class represents a node in the neural network.
################################################################################

package MachineLearning::NNNode;

use strict;
use warnings;
use Data::Dumper;
use Math::Random::MT::Perl;

use base 'Exporter';
our @EXPORT = qw(generateInnerNodes generateInputNodes generateOutputNodes);

my $debug = 0;

################################################################################
# NNNode constructor.
#
#   @param the id of the node.
#   @param the learning rate of the node. Pass in undef if not needed.
#    
#
#
#	@returns an NNNode object.
################################################################################
sub newNNNode {
    my $class = shift;
    my $id = shift;
    my $nu = shift; #the learning rate.
    
    $nu = ($nu) ? $nu : 0.01;   # default to 0.01 for the learning rate. 
                                # @TODO add nu methods tomorrow.

    my $self;

    #Used for node identification.
    $self->{NODE_ID} = $id;
    $self->{IS_BACK_PROPAGATABLE} = 1; #default to true;
    $self->{IS_FORWARD_PROPAGATABLE} = 1; #default to true;

    #Used for forward propagation
    $self->{OUTPUT}            = 0;    #Used to store the output of the NNNode.
    $self->{OUTPUT_IS_TO_DATE} = 0;    #Used to determine if the output is up-to-date.

    #Used to store the connected NNNodes to the node.
    $self->{IN_EDGES}        = ();     #A hash of the NNNodes and their keys.
    $self->{IN_EDGE_WEIGHTS} = ();     #A hash to store the weights the in edges.
    $self->{OUT_EDGES}       = ();     #A hash of the NNNodes and their keys.

    #Used for backwards propagation
    $self->{ERROR} = undef; #Used to store the calculated error of the NNNode.
    $self->{ERROR_IS_TO_DATE} = 0;
    
    $self->{NU} = $nu; 

    bless( $self, $class );

    return $self;
}

################################################################################
# Causes the output to be updated and calls the propagate method of each
# node connected with an out edge.
################################################################################
sub propagateForward {
    my $self = shift;
    
    $self->calculateOutput(); #calculate the output.

    #Only advance if the output is up-to-date and the node can be propagated onto.
    if ( $self->outputIsUpToDate() && $self->isForwardPropagatable() ) {
    
        #Propagate for all of the connected out edges.
        foreach my $key ( keys %{ $self->{OUT_EDGES} } ) {
            ${ $self->{OUT_EDGES} }{$key}->propagateForward();
        }
    }
    else {
    }
}

################################################################################
# Calculate the error.
#
#   @optional @param the expected value
################################################################################
sub calculateError {
        my $self = shift;
        my $t = shift; #the expected value
        
        $self->{ERROR_IS_TO_DATE} = 0; #The error is now out of date.
        
        my $error = 0;
        #apply the error calculation formula. 
        # If the node has output edges assume that it is an inner node.
        # calculate the error with the inner node.
        # Otherwise the node has no out edges then 
        # assume that it is an output and use the output node formula to 
        # calculate the error.
        my $size = (keys %{$self->{OUT_EDGES}});
        if($size > 0) {
            print "Node $self->{NODE_ID} is not an output node\n" if($debug);
            
            $error = $self->getOutput() * (1 - $self->getOutput());
            my $sum = 0;
            
            #The sum of the error and the weights multiplied from the connected outnodes.
            foreach my $key (keys %{$self->{OUT_EDGES}}) {
                my $weight = ${$self->{OUT_EDGES}}{$key}->getWeightForNodeID($self->getNodeID()); #Find the weight of this node in its out edge node.
                
                #See if the previuos node had been able to update it's error.
                if(${$self->{OUT_EDGES}}{$key}->errorIsUpToDate()) {
                    my $prevError = ${$self->{OUT_EDGES}}{$key}->getError();
                
                    $sum += ($weight * $prevError);
                }
                else {
                    print "\t$self->{NODE_ID}'s cannot be updated\n" if ($debug);
                    $self->{ERROR_IS_TO_DATE} = 0; #the error cannot be updated.
                    return; 
                }
            }
            
            $error *= $sum; #calculate the final error.
            
            $self->{ERROR} = $error;
            $self->{ERROR_IS_TO_DATE} = 1;
            
            print "\t$self->{NODE_ID}'s calculated error is $error\n" if ($debug);
        }
        else {
            print "Node $self->{NODE_ID} is an output node\n" if ($debug);
            $error = $self->getOutput() * (1 - $self->getOutput()) * ($t - $self->getOutput());
            
            $self->{ERROR} = $error;
            $self->{ERROR_IS_TO_DATE} = 1;
            
            print "\t$self->{NODE_ID}'s calculated error is $error\n" if ($debug);
        }
}

################################################################################
# The initial component to the backpropagation algorithm. Causes the 
# node to calculate its error and update its weight accordingly.
#
#   @optional @param the expected value
################################################################################
sub propagateBackwards {
    my $self = shift;
    my $t = shift;
    
    $self->calculateError($t); #Calculate the error for the node.
    
    if($self->{ERROR_IS_TO_DATE}) {
        $self->updateWeights(); #update the weights
        foreach my $key (keys %{$self->{IN_EDGES}}) {
            my $prevNode = ${$self->{IN_EDGES}}{$key};
            
            if($prevNode->isBackPropagatable()) {
                $prevNode->propagateBackwards();
            }
        }
    }
}

################################################################################
# Set or get the isBackPropagatable boolean var.
#
#   @optional @param a boolean value to set the propagatable property.
################################################################################
sub isBackPropagatable() {
    my $self = shift;
    my $isBP = shift;
    
    if(defined($isBP)) {
        $self->{IS_BACK_PROPAGATABLE} = $isBP;
    }
    else {
        return $self->{IS_BACK_PROPAGATABLE};
    }
}

################################################################################
#  Set or get the isForwardPropagatable boolean var.
#
#   @optional @param a boolean value to set the propagatable property.
################################################################################
sub isForwardPropagatable() {
    my $self = shift;
    my $isFP = shift;
    
    if(defined($isFP)) {
        $self->{IS_BACK_PROPAGATABLE} = $isFP;
    }
    else {
        return $self->{IS_BACK_PROPAGATABLE};
    }
}

################################################################################
# Set the id of the node. The id can be any value. But remember that
# all of the node ID's are compared as a string value.
#
#	@param the id of the node to use.
################################################################################
sub setNodeID {
    my $self = shift;

    $self->{NODE_ID} = shift;
}

################################################################################
# Get the id of the node.
#
#	@return the id of the node.
################################################################################
sub getNodeID {
    my $self = shift;

    return $self->{NODE_ID};
}

################################################################################
# Connect an NNNode to the in edge of this NNNode. Expects the NNNode object
# that is being connected to have a key already defined.
#
#	@param a NNNode object to add
################################################################################
sub addInEdge {
    my $self = shift;
    my $node = shift;

    ${ $self->{IN_EDGES} }{ $node->getNodeID() }        = $node;    #add the node to the in edge connections
    ${ $self->{IN_EDGE_WEIGHTS} }{ $node->getNodeID() } = 0;        #set the weight to zero.

    #	$self->generateInWeights(1);#Refresh all the weights (im just that lazy)
}

################################################################################
# Get the node connected to an in edge.
#
#   @param the id of the node connected to the in edge.
################################################################################
sub getInEdge {
    my $self = shift;
    my $id = shift;
    
    return ${ $self->{IN_EDGES} }{$id}; 
}

################################################################################
# Connect an NNNode to the out edge of this NNNode.
#
#	@param a NNNode object to add
################################################################################
sub addOutEdge {
    my $self = shift;
    my $node = shift;

    ${ $self->{OUT_EDGES} }{ $node->getNodeID() } = $node;
}

################################################################################
# Generate random weights for the in edges. All the weights of the in edges
# will be set to a random values between 0 and a specified number (non inclusive).
#
#	@param maximum value of weights (exclusive)
################################################################################
sub generateInWeights {
    my $self    = shift;
    my $maximum = shift;
    $maximum = $maximum * 2;

    my ($sec, $min, $day) = localtime(time);


    my @seed = (1...100);
    my $gen = Math::Random::MT::Perl->new(@seed);

    #Randomnize the weights
    foreach my $key ( keys %{ $self->{IN_EDGES} } ) {
        my $number = $gen->rand(101);    
        $number = ((($number / 100) * $maximum) - ($maximum / 2)); # -maximum <= rand <= maximum

        ${ $self->{IN_EDGE_WEIGHTS} }{$key} = $number;
    }
}

################################################################################
# Gets the output from the node.
################################################################################
sub getOutput {
    my $self = shift;

    return $self->{OUTPUT};

}

################################################################################
# Gets the weight belonging to a node ID connected via an in edge.
#
#   @param the id of the node to get the weight for.
################################################################################
sub getWeightForNodeID {
    my $self = shift;
    my $nodeID = shift;
    
    return ${$self->{IN_EDGE_WEIGHTS}}{$nodeID};
}

################################################################################
# Gets the error for the node.
################################################################################
sub getError {
    my $self = shift;
    
    return $self->{ERROR};
}

################################################################################
# Sets the output of the node.
#
#	@param the output to set the node to.
################################################################################
sub setOutput {
    my $self = shift;

    $self->{OUTPUT} = shift;
    $self->{OUTPUT_IS_TO_DATE} = 1;
}

################################################################################
# Set the target output of the node
#
#   @param the target value
################################################################################
# sub setTargetOutput {
    # my $self = shift;
    # my $target = shift;
    # 
    # $self->{TARGET_OUTPUT} = $target;
# }

################################################################################
# Get a boolean value if the output of the node is up-to-date xor not.
#
#	@return 1 xor 0 for true xor false.
################################################################################
sub outputIsUpToDate {
    my $self = shift;

    return $self->{OUTPUT_IS_TO_DATE};
}

################################################################################
# Get a boolean value if the error of the node is up-to-date xor not.
#
#	@return 1 xor 0 for true xor false.
################################################################################
sub errorIsUpToDate {
    my $self = shift;

    return $self->{ERROR_IS_TO_DATE};
}

################################################################################
# Recalculates the output. 
################################################################################
sub calculateOutput {
    my $self = shift;
    
    print "Node $self->{NODE_ID} is calculateing its output\n" if($debug);

    my $net = 0;
    foreach my $key ( keys %{ $self->{IN_EDGES} } ) {
        
        #If the out put has been up-dated in the previous node then 
        #continue summing the $net variable
        if(${$self->{IN_EDGES} }{$key}->outputIsUpToDate()) {
            my $output = ${ $self->{IN_EDGES} }{$key}->getOutput();
            print "\tThe output $output for node $key is TO-DATE\n" if($debug);
            $net += $output * 
                    ${ $self->{IN_EDGE_WEIGHTS} }{$key};
        }
        #Otherwise quit the method.
        else {
            my $output = ${ $self->{IN_EDGES} }{$key}->getOutput();
            print "\tThe output $output for node $key is OUT-OF-DATE.\n" if($debug);
            $self->{OUTPUT_IS_TO_DATE} = 0; #The output remains out of date until all of the in-nodes can update their output.
            return; 
        }
    }
    
    $self->{OUTPUT} = sigmoid($net);
    print "Node $self->{NODE_ID} output value is sigmoid($net) = $self->{OUTPUT}.\n" if($debug);

    $self->{OUTPUT_IS_TO_DATE} = 1;
}

################################################################################
# Represents the sigmoid function in the book.
#
#	@param a value 'x' to plug into the function.
################################################################################
sub sigmoid {
    my $x = shift;

    my $result = 1 / ( 1 + exp(-$x) );

    return $result;
}

################################################################################
# Updates the weights based on the error of the node. Assumes that the output
# of all of the connected nodes are up to date.
################################################################################
sub updateWeights {
    my $self = shift;
    
    foreach my $key (keys %{$self->{IN_EDGE_WEIGHTS}}) {
        my $x = ${$self->{IN_EDGES}}{$key}->getOutput(); #The output of the previous node.
        my $deltaWeight = $self->{NU} * $self->getError() * $x; #calculate the delta weight.
        
        my $weight = ${$self->{IN_EDGE_WEIGHTS}}{$key};
        $weight += $deltaWeight;
        
        ${$self->{IN_EDGE_WEIGHTS}}{$key} = $weight; #Update the new weight value.
    }
}

################################################################################
# Prints the node to the screen
################################################################################
sub toString {
    my $self = shift;

    print "--------------------\n";
    print "ID: $self->{NODE_ID}\n";
    print "Output: $self->{OUTPUT}\n";
    print "In Edges: (ID, W, OUT)\n";
    foreach my $k ( keys %{ $self->{IN_EDGES} } ) {
        my $node   = ${ $self->{IN_EDGES} }{$k};
        my $out    = $node->getOutput();
        my $weight = ${ $self->{IN_EDGE_WEIGHTS} }{$k};

        print "($k, $weight, $out)\t";

        if ( $node->getNodeID() ne $k ) {
            die "ID to Node ID missmatch\n";
        }
    }
    print "\n";
    print "Out Edges: (ID)\n";    #later display error value
    foreach my $k ( keys %{ $self->{OUT_EDGES} } ) {
        my $node = ${ $self->{OUT_EDGES} }{$k};

        print "($k)\t";

        if ( $node->getNodeID() ne $k ) {
            die "ID to Node ID missmatch\n";
        }
    }
    print "\n--------------------\n";

}

################################################################################
# Get or set the value of Nu.
################################################################################
sub Nu {
    my $self = shift;
    my $nu = shift;
    
    if(!defined($nu)) {
        return $self->{NU};
    }
    else {
        $self->{NU} = $nu;
    }
}

return 1;
