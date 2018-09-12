/*
 * ProgressBar.cpp
 *
 *  Created on: Sep 25, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */


#include "ProgressBar.h"
#include "llvm/Support/raw_ostream.h"
//#include <iostream>

using namespace CUDAAnalysis;
using namespace llvm;

void ProgressBar::printProgress()
{
	if (lastPrinted)
		return;

  ith++;
  double progress = 0;
  if (ith <= num)
    progress = (double)ith / (double)num;
  else
    progress = 1.0;

  int barWidth = 70;

  errs() << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      errs() << "=";
    else if (i == pos)
      errs() << ">";
    else
      errs() << " ";
  }
  int per = int(progress * 100.0);
  errs() << "] " << per << " %\r";
  if (per == 100)
  {
    errs() << "\n";
    lastPrinted = true;
    //errs() << " per == 100 " << "\n";
  }
  errs().flush();
}

void ProgressBar::printDone()
{
  ith = num;
  printProgress();
}
