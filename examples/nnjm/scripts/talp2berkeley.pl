#!/usr/bin/perl -w

## converts alignment format
## reverses order source/target and uses zero-index positions
## input format: 1-1 2-2 3-4 etc.
## output format: 0-0 1-1 3-2 etc.

use strict;

while (<>) {
    chomp;
    my @links = split(/ /);
    my $outLinks = "";
    my $separator = "";
    for my $link (@links) {
	my @pair = split(/\-/, $link);
	$outLinks .= $separator . ($pair[1] - 1) . "-" . ($pair[0] - 1);
	$separator = " ";
    }
    print "$outLinks\n";
}
