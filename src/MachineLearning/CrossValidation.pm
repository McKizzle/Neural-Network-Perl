################################################################################
# This class implements the nessessary components for cross-validation and
# collecting classification statistics.
################################################################################

package MachineLearning::CrossValidation;

use warnings;
use strict;
use Data::Dumper;
use List::Util qw(sum);

use base 'Exporter';
our @EXPORT = qw(fisher_yates_shuffle);

################################################################################
# Default Constuctor.
#
#	@param the number of folds.
#	@param an array reference of Classifiables as the training data.
#	@param a Classifier object to perform the cross fold validation on.
#
#	@return a CrossValidator object.
sub newCrossValidator {
    my $class        = shift;
    my $folds        = shift;
    my $trainingData = shift;
    my $classifier   = shift;

    my $self->{TRAINING_DATA} = undef;
    $self->{CLASSIFIER} = undef;
    $self->{FOLDS}      = undef;                                     #The number of folds the validator will use.
    $self->{CHUNKS}     = undef;                                     #An array to store the chunks.
    $self->{RESULTS}    = "Need to run the cross-validator first";
    $self->{CONFUSION_HASH}  = undef;                               #Stores the confusion matrix of all of the cross-validation runs.
    
    
    #This gets set when the validate method is called.
    $self->{SUB_TRAINING_DATA} = undef;
    $self->{SUB_TEST_DATA}     = undef;

    bless( $self, $class );

    #Set the classifier to undef or the
    #parameters that have been passed in.
    if ( defined($classifier) ) {
        $self->{CLASSIFIER} = $classifier;
    }

    #Set the training to undef or the
    #parameters that have been passed in.
    if ( defined($trainingData) ) {
        $self->{TRAINING_DATA} = $trainingData;
    }

    #Set the folds to undef or the
    #parameters that have been passed in.
    if ( defined($folds) ) {
        $self->{FOLDS} = $folds;

        #1 shuffle the array.
        fisher_yates_shuffle( $self->{TRAINING_DATA} );

        #2 Chunk up the array into $folds chunks.
        $self->chunkify();
    }

    return $self;
}

################################################################################
# Set the classifier. CrossValidator expects the classifier to already be
# set up properly. That is if any data (other than the test and training data)
# is needed.
#
#	@param a classifier that follows the classifier interface.
################################################################################
sub setClassifier {
    my $self       = shift;
    my $classifier = shift;

    $self->{CLASSIFIER} = $classifier;
}

################################################################################
# Set folds then shuffle and chunkify in the background.
#
#	@param the number of folds
################################################################################
sub setFolds {
    my $self  = shift;
    my $folds = shift;

    $self->{FOLDS} = $folds;

    if ( defined( $self->{TRAINING_DATA} ) ) {

        #Shuffle the array, then chunkify
        fisher_yates_shuffle( $self->{TRAINING_DATA} );
        $self->chunkify();
    }
}

################################################################################
# Sub run the cross-validator. Runs the classifier fold times on the
# training and test data
################################################################################
sub runCrossValidator {
    my $self   = shift;
    my $clsfr  = $self->{CLASSIFIER};
    my @chunks = @{ $self->{CHUNKS} };

    print "Running";

    #Run the classifier on each of the chunks
    # @TODO parallalalize the classification.
    for ( my $i = 0 ; $i < @chunks ; $i++ ) {
        my @testSet  = @{ $chunks[$i] };
        my @trainSet = ();

        for ( my $j = 0 ; $j < @chunks ; $j++ ) {
            if ( $j != $i ) {
                push( @trainSet, @{ $chunks[$j] } );
            }
        }

        $clsfr->setTrainingData( \@trainSet );
        $clsfr->setTestData( \@testSet );
        if($clsfr->isAnalysisMode()) {
            $clsfr->train(); #first train
            $clsfr->outputResultsToFile("results.csv", $i); #record progress to file
            
            $clsfr->classify(); #Now classify to get an accurate confusion matrix.
            
            #get the confusion hash from the results.
            my $confusionMatrix = $clsfr->getConfusionMatrix();
            if($self->{CONFUSION_HASH}) {
                sumMatricies($self->{CONFUSION_HASH}, $confusionMatrix);
            }
            else {
                $self->{CONFUSION_HASH} = $confusionMatrix;
            }
        }
        else {
            $clsfr->train();
            $clsfr->classify();
            
            #get the confusion hash from the results.
            my $confusionMatrix = $clsfr->getConfusionMatrix();
            if($self->{CONFUSION_HASH}) {
                sumMatricies($self->{CONFUSION_HASH}, $confusionMatrix);
            }
            else {
                $self->{CONFUSION_HASH} = $confusionMatrix;
            }
        }
        print ".";
    }
    print "\n";

    #collect the results and print the average accuracy and confusion
    # matrix to the screen.
    my $total   = 0; #this will be calculating by adding all of the values in the confusion matrix.
    my $correct = 0;

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

            if ( $actKey eq $predKey ) {
                $correct += ${ $confusionHash{$actKey} }{$predKey};
            }
            
            $total += ${ $confusionHash{$actKey} }{$predKey};
        }
        print "\n";
    }
    print "------------------------ Confusion Matrix --------------------------\n";
    print "\n";

    print "---------------------------- Accuracy ------------------------------\n";
    my $accuracy = ( $correct / $total ) * 100;
    print "The classifier had an accuracy of $accuracy% over the classification of $total objects\n";
    print "---------------------------- Accuracy ------------------------------\n";
    print "\n";
}

################################################################################
# Sum the results of two  confusion matricies. (NOTE TO SELF: These matricies 
# actually hashes.)
# 
#   @param matrix one. Matrix two is added into this matrix.
#   @param matrix two.
sub sumMatricies {
    my $matrixOne = shift;
    my $matrixTwo = shift;
    
    foreach my $actKey (keys %$matrixOne) {
        foreach my $predKey (keys %{${$matrixOne}{$actKey}}) {
            ${${$matrixOne}{$actKey}}{$predKey} = ${${$matrixOne}{$actKey}}{$predKey} + ${${$matrixTwo}{$actKey}}{$predKey};
        }
    }
}

################################################################################
# get folds
#
#	@return the number of folds
################################################################################
sub getFolds {
    my $self = shift;

    return $self->{FOLDS};
}

################################################################################
# This function will shuffle an array randomly.
#
#	@param an array reference
################################################################################
sub fisher_yates_shuffle {
    my $array = shift;
    my $i;
    for ( $i = @$array ; --$i ; ) {
        my $j = int rand( $i + 1 );
        next if $i == $j;
        @$array[ $i, $j ] = @$array[ $j, $i ];
    }
}

################################################################################
# The chunkify method will take the training data and divide it up into
# the specified number folds in the cross validator. If the set of data is not evenly
# divisable by the number of folds then the remaining elementes will be
# discarded. Once the candidate eliminator is implemented then come back and
# randomly distribute left overs.
sub chunkify {
    my $self      = shift;
    my @trainData = @{ $self->{TRAINING_DATA} };
    my $folds     = $self->{FOLDS};

    my $chunkSize = int( @trainData / $folds );

    #Initialize the chunks array
    my @chunks = ( (undef) x 10 );
    for ( my $i = 0 ; $i < @chunks ; $i++ ) {
        my @array = ();
        $chunks[$i] = \@array;
    }

    #Time to chop up veggies-n-squid.
    my $index = 0;
    for ( my $i = 0 ; $i < @trainData ; $i++ ) {

        #my $tmp = @trainData;
        $index = ( int( $i / $chunkSize ) < $folds ) ? int( $i / $chunkSize ) : last;
        push( @{ $chunks[$index] }, $trainData[$i] );
    }
    $self->{CHUNKS} = \@chunks;
}

return 1;

