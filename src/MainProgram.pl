################################################################################
# MainProgram is the core of the application and its job is to provide
# an interface that allows the user to load data and classify it
# using a desired classifier.
################################################################################

use strict;
use warnings;
use Data::Dumper;

#use Math::GSL; #used to calculate the factorial.
use MachineLearning::Classifiable;
use MachineLearning::CrossValidation;
use MachineLearning::NeuralNetwork;
use MachineLearning::NeuralNetworkClassifier;

$| = 1;    #forces a output flush when the application is running.

my @FOLDS = ("-folds", "-f");
my @CLASS_NAME = ("-class", "-c");
my @DATA_PATH = ("-data", "-d"); #The path for the data.
my @CROSS_VALIDATE = (1);
my @BIAS = ("-bias", "-b"); #The bias to default to.
my @LEARNING_RATE = ("-learning-rate", "-nu"); #The learning rate
my @NETWORK_STRUCTURE = ("-network-structure", "-ns"); #The structure of the network.
my @HELP = ("-help", "-h"); #Help
my @EPOCS = ("-epocs", "-e"); #The maximum number of epocs while training a network.
my @TRAINING_ANALYSIS_OUTPUT = ("-analysis", "-ao"); #The keyword that determines if the training results gets written to a file.y
my @WEIGHT_INIT_RANGE = ("-weight-init-range", "-wir"); #To set the weight randomnization range.
my @OPEN = ("-open", "-o"); #To open a file.
my @SAVE = ("-save", "-s"); #To save a file.


main(); #execute the main method.


sub main {
    my %argv = %{parseARGV()}; #convert the arguments into an easy to read form.
    
    #Get the essential inputs
    my $folds = (checkArg(\%argv, \@FOLDS)) ? ${$argv{checkArg(\%argv, \@FOLDS)}}[0] : 10;
    my $className = (checkArg(\%argv, \@CLASS_NAME)) ? ${$argv{checkArg(\%argv, \@CLASS_NAME)}}[0] : "class"; #used to determine which column is the class in teh csv file.
    my $dataPath = (checkArg(\%argv, \@DATA_PATH)) ? ${$argv{checkArg(\%argv, \@DATA_PATH)}}[0] : "data.csv";
    my $bias = (checkArg(\%argv, \@BIAS)) ? ${$argv{checkArg(\%argv, \@BIAS)}}[0] : 1;
    my $learningRate = (checkArg(\%argv, \@LEARNING_RATE)) ? ${$argv{checkArg(\%argv, \@LEARNING_RATE)}}[0] : 0.05;
    my $epocs = (checkArg(\%argv, \@EPOCS)) ? ${$argv{checkArg(\%argv, \@EPOCS)}}[0] : undef;
    my $networkStructure = (checkArg(\%argv, \@NETWORK_STRUCTURE)) ? $argv{checkArg(\%argv, \@NETWORK_STRUCTURE)} : undef; #get the array of the network structure.
    my $weightInitRange = (checkArg(\%argv, \@WEIGHT_INIT_RANGE)) ? $argv{checkArg(\%argv, \@WEIGHT_INIT_RANGE)} : 0.05; #default to 0.05.
    
    #Get the optional inputs
    my $trainingAnalysisOutput = (checkArg(\%argv, \@TRAINING_ANALYSIS_OUTPUT)) ? $argv{checkArg(\%argv, \@TRAINING_ANALYSIS_OUTPUT)} : "<<<>>>";
    my $open = (checkArg(\%argv, \@OPEN)) ? ${$argv{checkArg(\%argv, \@OPEN)}}[0] : undef;
    my $save = (checkArg(\%argv, \@SAVE)) ? ${$argv{checkArg(\%argv, \@SAVE)}}[0] : undef;
    
    #Get usage inputs
    my $help = (checkArg(\%argv, \@HELP)) ? 1 : 0;
    
    print "\n"x3;

    #Exit the application if eith epocs or networkStructure are not set.
    if($help) {
        help();
    }
    elsif(!$epocs && !$open) {
        print "'-e' or '-epocs' is needed\n";
        print "'-e' or '-epocs' needs a number following it\n";
    }
    elsif(!$networkStructure && !$open) { 
        print "'-ns' or '-network-structure' is needed\n";
        print "'-ns' or '-network-structure' needs at lease a single number following it.\n";
    }
    #Assumes that the data path points to test data and that a saved classifier is going to be loaded and then tested.
    elsif($open) {
        if($dataPath && $className) {
            print "Loading a network from a file to classify the user specified data.\n";
            #load the data
            my $data = csvToClassifiables( $dataPath, undef, $className );
            
            #load the saved classifier.
            my $classifier = LoadClassifierFromFile($open);
            $classifier->setTestData($data);
            
            #begin classifying the data.
            $classifier->classify();
            
            $classifier->printConfusionMatrix();
            
            print "\n\nThe accuracy of the neural network was ". $classifier->getResults(). ".\n";
        }
        else {
            error();
        }
    }
    #Assumes the data is the set of training data and writes a trained classifier to a file
    elsif($save) {
        if($dataPath && $bias && $learningRate && $epocs && $networkStructure || $trainingAnalysisOutput || $save) {
            print "Creating and training a network to save to a file. \n";
            #load the data
            my $data = csvToClassifiables( $dataPath, undef, $className );
            
            #Create a classifier.
            my $classifier = MachineLearning::NeuralNetworkClassifier->newNeuralNetworkClassifier();
            $classifier->setTrainingData($data);
            #Set the Network structure for the classifier.
            my @classVals = classValues($data);
            my @dimensions = $$data[0]->getDimensions();
            $classifier->neuralNetworkSettings($networkStructure, $bias, $learningRate, $weightInitRange, 
                \@classVals, \@dimensions);                   
            if($trainingAnalysisOutput !~ "<<<>>>") {
                $classifier->isAnalysisMode(1);
            }   
            else {
                $classifier->isAnalysisMode(0);
            }
            $classifier->epocs($epocs);
            
            #train the classifier
            $classifier->train();
            
            #write the classifier to a file
            $classifier->writeToFile($save);
            
            print "\n\nThe file $save contains the trained network\n";
        }
        else {
            error();
        }
    }
    #Perform a standard cross-validation with a neural network on a set of data.
    elsif($dataPath && $bias && $learningRate && $epocs && $networkStructure && $folds || $trainingAnalysisOutput || $save) {
        print "Creating and cros-validating on a user defined neural network. \n";
        
        #Load the data
        my $data = csvToClassifiables( $dataPath, undef, $className );
        
        #Initialize a CrossValidator.
        my $crossValidator = MachineLearning::CrossValidation->newCrossValidator($folds, $data);
        #Initialize a Classifier.
        my $classifier = MachineLearning::NeuralNetworkClassifier->newNeuralNetworkClassifier();
        #Set the Network structure for the classifier.
        my @classVals = classValues($data);
        my @dimensions = $$data[0]->getDimensions();
        $classifier->neuralNetworkSettings($networkStructure, $bias, $learningRate, $weightInitRange, 
            \@classVals, \@dimensions);                   
        if($trainingAnalysisOutput !~ "<<<>>>") {
            $classifier->isAnalysisMode(1);
        }   
        else {
            $classifier->isAnalysisMode(0);
        }
        $classifier->epocs($epocs);
        
        #set the classifier and perform the cross validation.
        $crossValidator->setClassifier($classifier); 
        $crossValidator->runCrossValidator(); #run the cross validator
    }
    else {
        error();
    }
    
    print "\n"x3;
}

################################################################################
# Parses the arguments sent in via the command line.
################################################################################
sub parseARGV {
    #parse the arguments.
    my %argsHash = ();

    for(my $i = 0; $i < @ARGV; $i++) {
        if($ARGV[$i] =~ m/-.+/) {
            # print "$ARGV[$i] is a parameter at ($i)\n";
            my @array = ();
            my $parameter = $ARGV[$i];
            
            my $j = $i + 1;
            # print "Getting values starting at: $j\n";
            while(($j < @ARGV) && ($ARGV[$j] !~ m/-.+/)) {
                # print "Adding $ARGV[$j] ($j) to $parameter\n"; 
                push(@array, $ARGV[$j]);
                $j++;
            }
            
            $argsHash{$parameter} = \@array;
        }
        else {}
    }
    
    return \%argsHash;
}

################################################################################
# Returns a boolean value determining if an argument was passed in.
#
#   @param an array refernce of the arguments.
#   @param an array of the different forms of the same command.
#
#   @return the command that is in the hash. otherwise return undef.
################################################################################
sub checkArg {
    my $argsHash = shift;
    my $commands = shift;
    
    my $key = undef;
    foreach(@$commands) {
        my $command = $_;
        
        if(${$argsHash}{$command}) {
            $key = $_;
        }
    }
    
    return $key;
}

################################################################################
# Print the usage information to the screen.
################################################################################
sub help {
    print "\n---------------------------USAGE-----------------------------------\n";
    print "NAME\n";
    print "\tMainProgram.pl";

    print "\nSYNOPSIS\n";
    print "\tMainProgram.pl [-data filepath] [-c class col name in csv] [-f folds]
\t\t[-bias bias] [-nu learning rate] [-ns network structure] 
\t\t[-e epocs]\n";
print "\tMainProgram.pl [-data filepath] [-c class col name in csv] [-bias bias] 
\t\t[-ns network structure] [-nu learning rate] [-e epocs] [-s file to save to]\n";
    print "\tMainProgram.pl [-data filepath] [-c class col name in csv] [-o network to load]\n";

    print "\nDESCRIPTION\n";
    print "\tThis application will classify a set of data using a neural network.
\tThe user will have the option of setting the fold count for the neural network. 
\tOnce done classifying the application will display the results and a confusion 
\tmatrix of the neural network.\n";

    print "\nOPTIONS\n";
    print "\t-A\n";
    print "\t\tThe program will look for the csv file 'data.csv' in the 
\t\texecutable's folder and use it. The application will default to 
\t\t'f=10' for the cross-validation if the '-f' option is not 
\t\tspecified. \n\n";

    print "\t-f, -folds=NUM\n";
    print "\t\tSet the number of folds to use for the cross-validator. This 
\t\tcan only be used if the '-data' option is used.\n\n";
		
    print "\t-c, -class=STRING\n";
    print "\t\tThe name of the column in the csv file that is the class.\n\n";
    
    print "\t-d, -data=PATH\n";
    print "\t\tUse this to set the data that will be used by the 
\t\tcross-validator. If no '-f' is specified then default to '-f=10'.\n\n";

    print "\t-b, -bias=NUM\n";
    print "\t\tThe default bias to use when initializing the neural network
\t\tclassifier.\n\n"; 

    print "\t-nu, -learning-rate=NUM\n";
    print "\t\tThe default learning rate to use when initializing the neural 
\t\tnetwork classifier.\n\n"; 

    print "\t-ns, -network-structure=NUM NUM ... NUM\n";
    print "\t\tA list of numbers representing the number of nodes per layer 
\t\tin the nueral network.\n\n"; 

    print "\t-e, -epocs=NUM\n";
    print "\t\tThe number of cycles to perform while training the network.\n\n"; 

    print "\t-wir, -weight-init-range=NUM\n";
    print "\t\tA value to set the initialization range of the network weights.\n\n"; 
    
    print "\t-o, -open=PATH\n";
    print "\t\tThe path of a file to load a classifier from.\n\n"; 
    
    print "\t-s, -save=PATH\n";
    print "\t\tThe path of a file to save a classifier to.\n\n"; 
    
    print "\t-ao, -analysis\n";
    print "\t\tWill simply print a csv file showing network accuracy as 
\t\t the training progresses.\n\n"; 

    print "\t-h, -help\n";
    print "\t\tPrint usage information to the screen.\n\n"; 

    print "-------------------------------------------------------------------\n\n";
}

################################################################################
# Print an error to the screen.
################################################################################
sub error {
    print "\n---------------------------ERROR-----------------------------------\n";
    print "Insuficient arguments. Pass \"-help\" in as a parameter for more info.\n";
    print "--------------------------------------------------------------------\n\n";
}


