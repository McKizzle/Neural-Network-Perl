################################################################################
# The NeuralNetworkClassifier class will classify data using a neural
# network. It's job is to store a network, train it, and then classify it.
# Not to be confused with the NeuralNetwork class. It's only job is to
# represent the network.
################################################################################

package MachineLearning::NeuralNetworkClassifier;

use strict;
use warnings;
use Data::Dumper;
$Data::Dumper::Purity = 1;
use Storable;
use MachineLearning::NNNode;
use MachineLearning::Classifiable;

use base 'Exporter';
our @EXPORT = qw(LoadClassifierFromFile);

my $debug = 0;

################################################################################
# Default constructor for the classifier. 
#
#	@optional @param training data as an array reference of Classifiable
#	 objects.
#	@optional @param test data as an array refernce of Classifiable
#	 objects.
#
# 	@returns a Candidate Elimination Classifier object.
################################################################################
sub newNeuralNetworkClassifier {
    my ( $class, $trainingData, $testData, ) = @_;

    my $self;
    
    $self->{TRAINED} = 0;
    
    $self->{IS_ANALYSIS_MODE} = 0; #if set to true the classifier will classify test data at the same time it is training.

    #Build an empty confusion "matrix";
    #In reality it will be a hash. This will enable the classifier to handle
    # other than boolean value classes.
    $self->{CONFUSION_HASH} = undef;
    $self->{RESULTS} = undef; #stores the percentage of correctly classified.
    
    $self->{NEURAL_NETWORK} = undef; #Stores the nueral network to classify with.
    $self->{MAX_EPOCS} = undef; # the maximum number of epocs to perform.
    $self->{ERRORS_PER_EPOC_TRAINING} = undef; #Stores the classification errors train dat.
    $self->{ERRORS_PER_EPOC_TEST} = undef; #Stores the classification errors test dat.
    $self->{WEIGHT_INIT_RANGE} = undef;
    
    #Variables to create a network.
    $self->{NET_STRUCT} = undef;
    $self->{NET_BIAS} = undef;
    $self->{NET_NU} = undef;
    $self->{NET_WEIGHT_INIT_RANGE} = undef;
    $self->{NET_CLASS_VALS} = undef; #also used to build the confusion matrix.
    $self->{NET_DIMENSIONS} = undef;
    
    #Set the training and test data to undef or the
    #parameters that have been passed in.
    if ( !defined($trainingData) ) {
        $self->{TRAINING_DAT} = undef;
    }
    else {
        $self->{TRAINING_DAT} = $trainingData;
    }
    if ( !defined($testData) ) {
        $self->{TEST_DAT} = undef;
    }
    else {
        $self->{TEST_DAT} = $testData;
    }

    bless( $self, $class );

    return $self;
}

################################################################################
# Sets all of the variables to properly create a neural network.
#
#   @param an array of numbers of length n. 
#       a_0 is the number of input nodes.
#       a_1 to a_{n-2} is the number of nodes in each inner layer.
#   @param the bias to set for all of the nodes.
#   @param the learning-rate to set for all of the nodes.
#   @param the weight initialization range.
#   @param an array reference of all of the possible class values
################################################################################
sub neuralNetworkSettings {
    my $self = shift; 
    my $networkStructure = shift;
    my $bias = shift;
    my $learningRate = shift; # Also refered to as Nu. (the fancy greek 'n')
    my $weightInitRange = shift; # The range to randomly init the weights to.
    my $clsVls = shift; #array ref to class vals
    my $dimensions = shift; #an array of the dimensions
    
    $self->{NET_STRUCT} = $networkStructure;
    $self->{NET_BIAS} = $bias;
    $self->{NET_NU} = $learningRate;
    $self->{NET_WEIGHT_INIT_RANGE} = $weightInitRange;
    $self->{NET_CLASS_VALS} = $clsVls;
    $self->{NET_DIMENSIONS} = $dimensions;
}

################################################################################
# create a neural network for the classifier. Expects neuralNetworkSettings to 
# be called at least once.
################################################################################
sub createNeuralNetwork {
    my $self = shift; 
    my $networkStructure = $self->{NET_STRUCT};
    my $bias = $self->{NET_BIAS};
    my $learningRate = $self->{NET_NU}; # Also refered to as Nu. (the fancy greek 'n')
    my $weightInitRange = $self->{NET_WEIGHT_INIT_RANGE}; # The range to randomly init the weights to.
    my $clsVls = $self->{NET_CLASS_VALS}; #array ref to class vals
    my $dimensions = $self->{NET_DIMENSIONS}; #an array of the dimensions
    $self->{WEIGHT_INIT_RANGE} = $weightInitRange;

    #First determine the number of class values
    my @classVals = @$clsVls;

    #Update the network structure to match the number of output nodes to the number of class vals.
    $$networkStructure[@$networkStructure - 1] = scalar(@classVals);

    #Create the network
    my $network = MachineLearning::NeuralNetwork->newNeuralNetwork(
                      $networkStructure, $bias, $learningRate
                  );
                  
    my @dimensions = @$dimensions;

    #Create the inputs for the input nodes.
    foreach(@dimensions) {
        $network->setInputValueForInputDimension($_, 0);
    }

    #Initialize all of the weights in the network
    $network->initializeWeights($weightInitRange);
    
    #Set the expected output values for the node.
    $network->setExpectedOutputValues(\@classVals);
    
    $self->{NEURAL_NETWORK} = $network;
    
}

################################################################################
# Set or get the analsis mode.
#
#   @optional @param a boolean value to set the mode.
################################################################################
sub isAnalysisMode {
    my $self = shift;
    my $val = shift;
    
    if(defined($val)) {
        $self->{IS_ANALYSIS_MODE} = $val;
    }
    else {
        return $self->{IS_ANALYSIS_MODE};
    }
}

################################################################################
# Trains the classifier.
################################################################################
sub train {
    my $self = shift;
    $self->createNeuralNetwork();
    my $network = $self->{NEURAL_NETWORK};
    
    print "\n\n\nDebugging train()\n" if($debug);
    print "\n\n\nRunning Analysis Mode\n" if($self->isAnalysisMode());
    
    my @errors = ();
    if($self->isAnalysisMode()) {
        my @testErrors = ();
        $self->{ERRORS_PER_EPOC_TEST} = \@testErrors;
    }
    
    foreach((1...$self->{MAX_EPOCS})) {
        print "\n\n=======================================Epoc $_<<<<\n" if($debug || $self->isAnalysisMode());
        
        my $incorrect = 0;
        my $total = @{$self->{TRAINING_DAT}};
        
        #Go through all of the classifiables.
        foreach(@{$self->{TRAINING_DAT}}) {
                my $aClsbl = $_;
                
                my $classVal = $aClsbl->getClass();
                
                #Set the input values for the network.
                foreach($aClsbl->getDimensions()) {
                    my $dimen = $_;
                    my $dimenVal = $aClsbl->getDimenValue($dimen);
                    
                    $network->setInputValueForInputDimension($dimen, $dimenVal);
                }
                
                $network->propagateForward();
                my $classValPred = $network->predictClass();
                $network->propagateBackwards($classVal);
                
                $incorrect = ($classValPred != $classVal) ? $incorrect + 1: $incorrect;
        }
        
        my $percent = ($incorrect / $total) * 100;
        push(@errors, $percent);
        print "$percent% where classified incorrectly in the training data.\n" if($debug || $self->isAnalysisMode());
        
        #IF in anaylsis mode then run the classify method.
        if($self->isAnalysisMode()) {
            $self->classify(); #classify the training data.
        }
    }
    
    $self->{ERRORS_PER_EPOC_TRAINING} = \@errors; #store the errors
}

################################################################################
# Classifies the data once the classifier has been trained.
################################################################################
sub classify {
    my $self = shift;
    
    print "Debugging classify()\n" if($debug);
    
    my $network = $self->{NEURAL_NETWORK};
    
    my $errors = $self->{ERRORS_PER_EPOC_TEST} if($self->isAnalysisMode()); #Stores the errors as an array. This is to write a csv file of the results.
    
    $self->createConfusionMatrix(); #reset the confusion matrix before classifying the data. 
    
    #to tally the incorrectly classified.
    my $incorrect = 0;
    my $total = @{$self->{TEST_DAT}};
    
    #Go through all of the classifiables.
    foreach(@{$self->{TEST_DAT}}) {
        my $aClsbl = $_;

        my $classVal = $aClsbl->getClass();

        #Set the input values for the network.
        foreach($aClsbl->getDimensions()) {
            my $dimen = $_;
            my $dimenVal = $aClsbl->getDimenValue($dimen);
            
            $network->setInputValueForInputDimension($dimen, $dimenVal);
        }

        #propagate forward through the network so that we can classify the classifiable object.
        $network->propagateForward();
        my $classValPred = $network->predictClass();
        
        #update the confusion matrix accordinly.
        my $confusionHash = $self->{CONFUSION_HASH};
        ${${$confusionHash}{$classVal}}{$classValPred}++; #increment the proper index of the confusion "matrix".

        $incorrect = ($classValPred != $classVal) ? $incorrect + 1: $incorrect; #determine if the classification was correct.
    }
    
    #calculate percentages
    my $percent = ($incorrect / $total) * 100;
    my $percentCorrect = 100 - $percent;
    
    push(@$errors, $percent) if($self->isAnalysisMode()); #store the classification results such that they can be written to a file if in analysis mode.
    print "$percent% where classified incorrectly in the test data.\n" if($debug || $self->isAnalysisMode());
    
    $self->{RESULTS} = $percentCorrect; #store the percentage correct.
}

################################################################################
# Set the training data for the classifier.
#
#	@param an array reference of classifiable objects.
################################################################################
sub setTrainingData {
    my $self         = shift;
    my $trainingData = shift;

    $self->{TRAINING_DAT} = $trainingData;
}

################################################################################
# Set or get the number of epocs to perform while training the network.
#
#	@param epoc count
################################################################################
sub epocs {
    my $self = shift;
    my $epocs = shift;
    
    if($epocs) {
        $self->{MAX_EPOCS} = $epocs;
    }
    else {
        return $self->{MAX_EPOCS};
    }
}

################################################################################
# Gets the training data from the classifier
#	@return an array reference of the training data.
################################################################################
sub getTrainingData {
    my $self = shift;

    return $self->{TRAINING_DAT};
}

################################################################################
# Set the test data for the classifier.
#
#	@param an array reference of test data.
################################################################################
sub setTestData {
    my $self     = shift;
    my $testData = shift;

    $self->{TEST_DAT} = $testData;
}

################################################################################
# Get the test data from the classifier.
#
#	@return an array reference of test data.
################################################################################
sub getTestData {
    my $self = shift;

    return $self->{TEST_DAT};
}

################################################################################
# Returns the classifier results
################################################################################
sub getResults {
    my $self = shift;

    if ( !defined( $self->{RESULTS} ) ) {
        return "Need to classify the test data first";
    }
    else {
        return $self->{RESULTS};
    }
}

################################################################################
# creates a new confusion matrix for the classifier.
################################################################################
sub createConfusionMatrix {
    my $self = shift;
    my @classVals = @{$self->{NET_CLASS_VALS}}; #Different class values.
	
	#initialize the 2D results hash with 0's
	my %confusion = ();
	foreach(@classVals) {
		my $actKey = $_;
		my %predHash = ();
		foreach(@classVals) {
			my $predKey = $_;
			$predHash{$predKey} = 0;
		}
		$confusion{$actKey} = \%predHash;
	}
    
    $self->{CONFUSION_HASH} = \%confusion;
}

################################################################################
# Returns a copy of the confusion hash.
#
#	@return a copy of the confusion hash.
################################################################################
sub getConfusionMatrix {
    my $self = shift;
    return $self->{CONFUSION_HASH};
}

################################################################################
# Prints the confusion matrix to the screen
sub printConfusionMatrix {
    my $self = shift;
    my %confusionHash = %{ $self->{CONFUSION_HASH} };

    #print the resulting confusion matrix to the screen
    #maybe this should be printed as a csv? 
    my @keys = ( keys %confusionHash );
    print "\n";
    print "------------------------ Confusion Matrix --------------------------\n";
    print "The rows are the actual class.\n";
    print "The cols are the predicted class.\n";
    print "\t\t";
    foreach(@keys) {
        print "Class: $_\t";
    }
    print "\n";
    foreach (@keys) {
        my $actKey = $_;
        print "\n";
        print "Class: $actKey \t";
        foreach (@keys) {
            my $predKey = $_;
            print ${ $confusionHash{$actKey} }{$predKey} . "\t\t";
        }
        print "\n";
    }
    print "------------------------ Confusion Matrix --------------------------\n";
    print "\n";
}

################################################################################
# Outputs the results of the classifier when it ran in analysis mode.
#
#   @param the file name and file path together (eg) path/filename.
#   @param the trial count.
################################################################################
sub outputResultsToFile {
    my $self = shift;
    my $filePath = shift;
    my $trial = shift;
    
    open( OUT, ">>$filePath" ) || die "Failed to read $filePath or create $filePath. Check path. \n--------------------------------------------------------------------\n\n";
        print OUT "Trial $trial\n";
        print OUT "Training Data,";
        foreach(@{$self->{ERRORS_PER_EPOC_TRAINING}}) {
            print OUT $_;
            print OUT ",";
        }
        print OUT "\n";
        print OUT "Test Data,";
        foreach(@{$self->{ERRORS_PER_EPOC_TEST}}) {
            print OUT $_;
            print OUT ",";
        }
        print OUT "\n\n";
        
    close( OUT );
}

################################################################################
# Writes a classifier to a file.
#
#   @param the path of the file.
#
################################################################################
sub writeToFile {
    my $self = shift;
    my $filePath = shift;
    
    # open( OUT, ">$filePath" ) || die "Failed to read $filePath or create $filePath. Check path. \n--------------------------------------------------------------------\n\n";
    # print OUT Data::Dumper->Dump([$self], ['*self']);
    # close(OUT); 
    
    store $self, $filePath; 
}
 
################################################################################
# Loads a classifier from a file. 
#
#   @param the path of the file.
#
#   @return a NeuralNetworkClassifier object.
################################################################################
sub LoadClassifierFromFile {
    my $filePath = shift;
    
    my $neuralNetworkClassifier = retrieve($filePath);
    
    return $neuralNetworkClassifier;
}

return 1;
