#!/usr/bin/perl
use warnings;
use strict;
use Text::CSV;




sub intersectLines {
#working subroutine. thanks to the original poster.
    my( $ax, $ay, $bx, $by, $cx, $cy, $dx, $dy )= @_;
    my @rval=0;

    my $d1=($ax-$bx)*($cy-$dy);
    my $d2=($ay-$by)*($cx-$dx);

    my $dp = $d1 - $d2;
    my $dq = $d2 - $d1;

    if($dp!=0 && $dq!=0) {
       my $p = ( ($by-$dy)*($cx-$dx) - ($bx-$dx)*($cy-$dy) ) / $dp;
       my $q = ( ($dy-$by)*($ax-$bx) - ($dx-$bx)*($ay-$by) ) / $dq;
      if($p>0 && $p<=1 && $q>0 && $q<=1) {
         my $px= $p*$ax + (1-$p)*$bx;
         my $py= $p*$ay + (1-$p)*$by;
         @rval=($px, $py);
      }
   }

   return(@rval);
}

#returns 0 if no collision; 1 if collision
#params
#center of rectangle
#start point of line
#end point of line
sub testCollision {
  my ($cX, $cY, $aX, $aY, $bX, $bY) = @_;
  my @r;
  @r = intersectLines($cX-0.5, $cY-0.5, $cX-0.5, $cY+0.5, $aX, $aY, $bX, $bY);
  if (defined $r[1]) {return 1};
  @r = intersectLines($cX-0.5, $cY+0.5, $cX+0.5, $cY+0.5, $aX, $aY, $bX, $bY);
  if (defined $r[1]) {return 1};
  @r = intersectLines($cX+0.5, $cY+0.5, $cX+0.5, $cY-0.5, $aX, $aY, $bX, $bY);
  if (defined $r[1]) {return 1};
  @r = intersectLines($cX+0.5, $cY-0.5, $cX-0.5, $cY-0.5, $aX, $aY, $bX, $bY);
  if (defined $r[1]) {return 1};
  return 0;
}

# my @r = intersectLines(1,3,1,0,0,1,4,1);
# print "$r[0]/$r[1]\n";
# @r = intersectLines(0,1,4,1,1,0,3,0);
# if (defined $r[1]) {
#   print "$r[0]/$r[1]\n";
# } else {
#   print "no intersection\n";
# }
# @r = intersectLines(1,3,1,0,2,2,4,3);
# if (defined $r[1]) {
#   print "$r[0]/$r[1]\n";
# } else {
#   print "no intersection\n";
# }
#
# exit;












my ($trackFileName, $tripFileName, $outputFileName) = @ARGV;
my $line;
my @track;
my @rTrack;
my @obstacles;
my @finish;
my $cntObs=0;
my $cntFin=0;
my $finishedReached=0;
my $dimTrackX;
my $dimTrackY;
my @trips;
my $iTrip=0;
my $maxX=0;
my $maxY=0;
my $i=0;
my $j = 0;
my $curX;
my $curY;
my $prevX;
my $prevY;
my $prevSpeedX;
my $prevSpeedY;
my $curSpeedX;
my $curSpeedY;
my @wrongPoints;
my $prevWasGrass;

# read track file
open my $trackFile, $trackFileName or die "Could not open $trackFileName: $!";
$i=0;
while($line = <$trackFile>)  {
  chomp $line;
  $j=0;
  foreach my $char (split //, $line) {
    $rTrack[$j][$i] = $char;
    ++$j;
  }
  ++$i;
}
$dimTrackX=$j;
$dimTrackY=$i;
close $trackFile;

for ($i = 0; $i < $dimTrackX; ++$i) {
  for ($j = 0; $j < $dimTrackY; ++$j) {
    $track[$i][$j] = $rTrack[$i][$dimTrackY - $j - 1];
    if ($track[$i][$j] eq "O") {
      my @hlp = ();
      $obstacles[$cntObs][0] = $i;
      $obstacles[$cntObs][1] = $j;
      $cntObs++;
    }
    if ($track[$i][$j] eq "F") {
      my @hlp = ();
      $finish[$cntFin][0] = $i;
      $finish[$cntFin][1] = $j;
      $cntFin++;
    }
  }
}

# read trip file
my $csv = Text::CSV->new({ sep_char => ',' });
open(my $tripData, '<', $tripFileName) or die "Could not open '$tripFileName' $!\n";
$i=0;
FILEREAD: while ($line = <$tripData>) {
  chomp $line;

  if ($csv->parse($line)) {
    @{$trips[$iTrip][$i]} = $csv->fields();
    if ($maxX < $trips[$iTrip][$i][0]) { $maxX = $trips[$iTrip][$i][0]; }
    if ($maxY < $trips[$iTrip][$i][1]) { $maxY = $trips[$iTrip][$i][1]; }

    # check if speed is correct
    $curX = $trips[$iTrip][$i][0];
    $curY = $trips[$iTrip][$i][1];
    if ($i == 0) {
      if ($track[$curX][$curY] ne "S") {
        #error -> should be at the start line
        push(@wrongPoints, ($trips[$iTrip][$i]));
      }
      $prevWasGrass = 0;
      $prevX = $curX;
      $prevY = $curY;
      $prevSpeedX = 0;
      $prevSpeedY = 0;
    }
    $curSpeedX = $curX - $prevX;
    $curSpeedY = $curY - $prevY;

    if (($prevX + $prevSpeedX - 1 > $curX) || ($prevX + $prevSpeedX + 1 < $curX) || ($prevY + $prevSpeedY - 1 > $curY) || ($prevY + $prevSpeedY + 1 < $curY)) {
      # error
      push(@wrongPoints, ($trips[$iTrip][$i]));
    }
    if ($prevWasGrass == 1) {
      if ((abs($prevSpeedX) > 1) && (abs($curSpeedX) >= abs($prevSpeedX))) {
        # error
        push(@wrongPoints, ($trips[$iTrip][$i]));
      }
      if ((abs($prevSpeedY) > 1) && (abs($curSpeedY) >= abs($prevSpeedY))) {
        # error
        push(@wrongPoints, ($trips[$iTrip][$i]));
      }
    }

    #test whether an obstacle is crossed
    foreach my $o (@obstacles) {
      if (testCollision(@$o[0],@$o[1],$prevX,$prevY,$curX,$curY) == 1) {
        # error
        push(@wrongPoints, ($trips[$iTrip][$i]));
        last;
      }
    }

    # test whether the finish line was crossed
    foreach my $f (@finish) {
      if (testCollision(@$f[0],@$f[1],$prevX,$prevY,$curX,$curY) == 1) {
        #finished line crossed -> we stop here
        $finishedReached = 1;
        last FILEREAD;
      }
    }

    # prepare next iteration
    if (($trips[$iTrip][$i][0] > $dimTrackX) || ($trips[$iTrip][$i][0] < 0) || ($trips[$iTrip][$i][1] > $dimTrackY) || ($trips[$iTrip][$i][1] < 0) || ($track[$trips[$iTrip][$i][0]][$trips[$iTrip][$i][1]] eq "G")) {
      $prevWasGrass = 1;
    } else {
      $prevWasGrass = 0;
    }
    $prevX = $trips[$iTrip][$i][0];
    $prevY = $trips[$iTrip][$i][1];
    $prevSpeedX = $curSpeedX;
    $prevSpeedY = $curSpeedY;
  } else {
    warn "Line could not be parsed: $line\n";
  }
  ++$i;
}
#test whether the finish line was crossed
if ($finishedReached == 0) {
  #error -> should be at the finish line
  push(@wrongPoints, ($trips[$iTrip][$i-1]));
}


close $tripData;

# and now write the latex file
open(FH, '>', $outputFileName) or die $!;
print FH "
% tikzpic.tex
\\documentclass[crop,tikz]{standalone}%
%\\usetikzlibrary{...}% tikz package already loaded by 'tikz' option
\\begin{document}
\\begin{tikzpicture}
  \\definecolor{colGrass}{RGB}{52, 194, 0};
  \\definecolor{colFinish}{RGB}{128, 164, 255};
  \\definecolor{colTrip}{RGB}{0, 0, 0};
  \\definecolor{colError}{rgb}{0.47,0.047,0.114};
  \\definecolor{aitGrey}{rgb}{0.72,0.72,0.72};
  \\definecolor{aitLightGrey}{rgb}{0.86,0.86,0.86};
  \\definecolor{aitDarkGrey}{rgb}{0.36,0.36,0.36};
  \\definecolor{aitRed}{rgb}{0.47,0.047,0.114};
  \\definecolor{aitTurquoise}{rgb}{0,0.596,0.6};
  \\definecolor{aitLightViolette}{rgb}{0.909,0.560,1.000};
  \\definecolor{aitViolette}{rgb}{0.278,0.058,0.317};
  \\definecolor{aitLightTurquoise}{rgb}{0,0.798,0.8};\n";

for ($i = 0; $i < $dimTrackX; ++$i) {
  for ($j = 0; $j < $dimTrackY; ++$j) {
    print FH "  \\fill[color=";
    if ($track[$i][$j] eq "T") {
      print FH "aitLightGrey";
    } elsif ($track[$i][$j] eq "S") {
      print FH "aitGrey";
    } elsif ($track[$i][$j] eq "G") {
      print FH "colGrass";
    } elsif ($track[$i][$j] eq "F") {
      print FH "colFinish";
    } else {
      print FH "aitDarkGrey";
    }
    print FH "] (" . ($i-0.51) . "," . ($j-0.51) . ") rectangle +(1.01,1.01);\n";
  }
}

print FH "  \\draw[step=1.0,aitGrey,thin,dashed] (-2,-2) grid (".($dimTrackX+1).",".($dimTrackY+1).");
  \\draw[color=colTrip,line width=3pt,mark=x,mark size=8pt] plot coordinates { ";
foreach my $trip (@trips) {
  foreach my $point (@$trip) {
    print FH "(" . @$point[0] . "," . @$point[1] . ") ";
  }
}
print FH "};\n";
$i=0;
foreach my $trip (@trips) {
  foreach my $point (@$trip) {
    print FH "  \\node[anchor=south west] at (" . @$point[0] . "," . @$point[1] . ") {\\Large " . $i++ . "};\n";
  }
}


# and now mark all errors
foreach my $error (@wrongPoints) {
  print FH "  \\draw[color=colError,line width=4pt] (@$error[0],@$error[1]) circle (12pt);\n"
}

# label the grid
for ($i = 0; $i < $dimTrackX; ++$i) {
  print FH "  \\node at ($i,-1) {\\LARGE $i};\n";
}
for ($j = 0; $j < $dimTrackY; ++$j) {
  print FH "  \\node at (-1,$j) {\\LARGE $j};\n";
}

print FH "\\end{tikzpicture}
\\end{document}";

close(FH);


