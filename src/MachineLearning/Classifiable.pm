################################################################################
# The Classifiable class represents a classifiable object in memory. This is
# usefull for storing data from CSV files. Classifiers that are implemented
# will expect Classifiable objects when classifying data.
################################################################################

package MachineLearning::Classifiable;

use warnings;
use strict;
use Data::Dumper;
use Carp;

#Export class functions
use base 'Exporter';
our @EXPORT = qw(classValues csvToClassifiables isSameClass mostCommonClass dimensionValues);

################################################################################
# csvToClassifiables will return an array of classifiable objects. The
# by default assumes the column delimeter is ','.
# @calls new() method in Classifiable.pm
#
#	@param The path of the csv file to be opened.
#	@optional @param The type of delimeter is used to seperate columns.
#	@optional @param The the name of the column that the will be the class.
#
#	@note Second param needs to be set to undef if you are passing in a value to represent
#		the column that is the class in the data.
#
#	@returns an array reference of classifiable objects
sub csvToClassifiables {

    # Get all the parameters
    my ( $filePath, $colDelimeter, $classCol ) = @_;

    $colDelimeter = ( !defined($colDelimeter) ) ? ','     : $colDelimeter;
    $classCol     = ( !defined($classCol) )     ? 'class' : $classCol;

    open( IN, "<$filePath" ) || die "Failed to read $filePath. Check path. \n--------------------------------------------------------------------\n\n";

    my $firstLine = <IN>;
    my @columnNames = split( $colDelimeter, $firstLine );

    #Get rid of windows new line characters
    for ( my $i = 0 ; $i < @columnNames ; $i++ ) {
        $columnNames[$i] =~ s/\r?\n?//g;    #get rid of the new line and carriage return junk!!
    }

    my @classifiables = ();
    while ( my @line = split( $colDelimeter, <IN> ) ) {
        chomp(@line);                       #remove any \n's

        my %properties = ();
        my $class;

        for ( my $i = 0 ; $i < @line ; $i++ ) {
            $line[$i] =~ s/\r?\n?//g;       #get rid of the junk

            if ( $columnNames[$i] !~ m/$classCol/i ) {
                $properties{ $columnNames[$i] } = $line[$i];
            }
            else {
                $class = $line[$i];
            }
        }

        my $aClassifiable = MachineLearning::Classifiable->new( \%properties, $class );
        push( @classifiables, $aClassifiable );

    }    #END WHILE LOOP

    close(IN);    #Close the handle.

    return \@classifiables;
}

################################################################################
# Default constructor for a Classifiable object.
#
#	@param a hash containing all the properties (dimensions) for an object.
#	@param a variable for the class of the object.
#
#	@return a Classifiable object.
sub new {
    my $class = shift;                 #get the class
    my $self->{PROPERTIES} = shift;    #get the properties
    $self->{CLASS} = shift;            #get the class

    #Bless the object.
    bless( $self, $class );

    return $self;
}

################################################################################
#Returns the properties of the Classifiable object.
#
#	@return a hash of the fruit object properties.
sub getProperties {
    my $self = shift;
    return $self->{PROPERTIES};
}

################################################################################
#Get the value for the specified dimension of the object.
#
#	@param the dimension in question.
#
#	@return the value of the dimension.
sub getDimenValue {
    my $self      = shift;
    my $dimension = shift;

    return ${ $self->{PROPERTIES} }{$dimension};

}

################################################################################
# Returns the class of the Classifiable object
#
#	@return fruit class
sub getClass {
    my $self = shift;
    return $self->{CLASS};
}

################################################################################
# Return the dimensions for the Classifiable object
#
#	@return an array of the dimensions in the object
sub getDimensions {
    my $self = shift;
    return ( keys %{ $self->{PROPERTIES} } );
}

################################################################################
# A function that sees if an array of classifiables belong
# to the same class.
#
#	@param an array refernce of a set of classifiable
#	 objects
#
#	@return 1 if they are all the same class. 0
#		otherwise.
#
sub isSameClass {
    my $classifiables = shift;

    my $isSame    = 1;
    my $prevClass = ${$classifiables}[0]->getClass();
    for ( my $i = 1 ; $i < @{$classifiables} ; $i++ ) {
        my $currClass = ${$classifiables}[$i]->getClass();

        if ( $currClass != $prevClass ) {
            $isSame = 0;
            last;
        }
        $prevClass = $currClass;
    }

    return $isSame;
}

################################################################################
# This function returns the most common class in a set of
# classifiable objects.
#
#	@param an array refernce of a set of classifiable
#	 objects
#
#	@return the most common class.
sub mostCommonClass {
    my $classifiables = shift;

    my %classTally = ();
    foreach ( @{$classifiables} ) {
        my $aClassifiable = $_;
        my $class         = $aClassifiable->getClass();

        if ( !defined( $classTally{$class} ) ) {
            $classTally{$class} = 1;
        }
        else {
            $classTally{$class}++;
        }
    }

    my $maxClass   = undef;
    my $classCount = undef;
    foreach my $key ( keys %classTally ) {
        if ( !defined($maxClass) ) {
            $maxClass   = $key;
            $classCount = $classTally{$key};
        }
        elsif ( $classCount < $classTally{$key} ) {
            $maxClass   = $key;
            $classCount = $classTally{$key};
        }
    }

    return $maxClass;
}

################################################################################
# Returns all of the possible values for a dimension in a set of classifiables
#
#	@param the dimension in question.
# 	@param references to arrays of a set of classifiables.
#
#	@return all the possible different values of the dimension as an
#		array refrence.
################################################################################
sub dimensionValues {
    my $dimen = shift;
    my @dimValues = map { $_->getDimenValue($dimen) } @{ shift @_ };    #extract the column for the dimension

    my %encountered = ();
    my @result = grep { !$encountered{$_}++ } @dimValues;

    return \@result;
}

################################################################################
# Returns all of the possible class values in a set of classifiables
#
# 	@param references to arrays of a set of classifiables.
#
#	@return all the possible different values of the class in the set of
#		of classifiables as an array.
################################################################################
sub classValues {
    my @classValues = map { $_->getClass() } @{ shift @_ };    #extract the classes from teh classifiables.

    my %encountered = ();
    my @result = grep { !$encountered{$_}++ } @classValues;

    return @result;
}

#All packages must return 1;
return 1;

