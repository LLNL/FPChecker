/*
 * ProgressBar.h
 *
 *  Created on: Sep 25, 2015
 *      Author: Ignacio Laguna
 *     Contact: ilaguna@llnl.gov
 */
#ifndef CODE_SRC_PROGRESSBAR_H_
#define CODE_SRC_PROGRESSBAR_H_

namespace CUDAAnalysis {

class ProgressBar
{
private:
  unsigned num; // number of items or iterations
  unsigned ith; //
  bool lastPrinted;
public:
  ProgressBar(unsigned n) : num(n), ith(0), lastPrinted(false) {};
  void printProgress();
  void printDone();
};

}

#endif /* CODE_SRC_PROGRESSBAR_H_ */
