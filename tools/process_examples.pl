#!/usr/bin/perl

open(MYFILE, 'examples.csv');
open(OUT, ">examples.csv.out");
open(IN, ">examples.csv.in");


while(<MYFILE>){
    chomp;
    s/\?/-1/g; # processes missing variables
    s/\r//g;
    
    @words = split(" ", $_);
    
    $N = $#words - 1;
    
#    print "$#words \n";
    
    for my $i (0 .. $N){
	print IN "@words[$i] ";
    }
    print IN "\n";
    
    print OUT "@words[$#words] ";

    print OUT "\n";
}


close(MYFILE);
close(OUT);
close(IN);
