#!/usr/bin/perl -w

## input: file with loglikelihood per instance
## input: text file with one sentence per line (no start sentence or end sentence marker)
## output (stdout): loglikelihood per sentence

use strict;

if (@ARGV != 2) {
    print STDERR "Args: <loglikelihood per instance file> <sentence file>\n";
    exit(1);
}

my ($instanceFile, $sentenceFile) = @ARGV;

open(INSTANCE, "zcat -f $instanceFile |") or die("Cannot open file $instanceFile: $!\n");
open(SENTENCE, "zcat -f $sentenceFile |") or die("Cannot open file $sentenceFile: $!\n");

while (my $sentence = <SENTENCE>) {
    chomp $sentence;
    my @sentenceWords = split(/\s+/, $sentence);
    my $sentenceLogLikelihood = 0;
    for my $word (@sentenceWords) {
	my $instanceLogLikelihood = <INSTANCE>;
	if (! $instanceLogLikelihood) {
	    print STDERR "Mismatched number of words in $instanceFile and $sentenceFile\n";
	    exit(1);
	}
	chomp $instanceLogLikelihood;
	$sentenceLogLikelihood += $instanceLogLikelihood;
    }
    # end of sentence
    my $instanceLogLikelihood = <INSTANCE>;
    if (! $instanceLogLikelihood) {
	print STDERR "Mismatched number of words in $instanceFile and $sentenceFile\n";
	exit(1);
    }
    chomp $instanceLogLikelihood;
    $sentenceLogLikelihood += $instanceLogLikelihood;
    print STDOUT "$sentenceLogLikelihood\n";
}
