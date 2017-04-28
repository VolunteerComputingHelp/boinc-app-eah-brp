/*----------------------------------------------------------------------------
 *
 * File Name: rngmed.c
 * Authors:  B. Machenschalk
 *
 *
 * Description: efficient computation of a running median
 *
 * This implements https://dcc.ligo.org/LIGO-T030168/public
 * Soumya Mohanty "Efficient Algorithm for Computing a Running Median" (2003/4)
 *
 *----------------------------------------------------------------------------
 */


/* compilation flags */
#define NEWCHECKPOINT /* add a checkpoint for the median */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "rngmed.h"


/*Used in running qsort*/
static int rngmed_qsortindex(const void *elem1, const void *elem2)
{
  struct qsnode
  {
    float value;
    unsigned int index;
  };

  const struct qsnode *A = (const struct qsnode*) elem1;
  const struct qsnode *B = (const struct qsnode*) elem2;

  if (B->value > A->value)
    return -1;
  else if (A->value > B->value)
    return 1;
  else if (A->index > B->index)
    return -1;
  else
    return 1;
}


void rngmed(const float *input, const unsigned int length, const unsigned int bsize, float *medians)
{
  /* a single "node"
     lesser  points to the next node with less or equal value
     greater points to the next node with greater or equal value
     an index == blocksize is an end marker
  */
  struct node
  {
    float value;
    unsigned int lesser;
    unsigned int greater;
  };

  /* a node of the quicksort array */
  struct qsnode
  {
    float value;
    unsigned int index;
  };

  const unsigned int nil = bsize;     /* invalid index used as end marker */
  const int isodd = bsize&1;          /* bsize is odd = median is a single element */
  struct node* nodes;                 /* array of nodes, will be of size blocksize */
  struct qsnode* qsnodes;             /* array of indices for initial qsort */
  unsigned int* checkpts;             /* array of checkpoints */
  unsigned int  ncheckpts,stepchkpts; /* checkpoints: number and distance between */
  unsigned int  oldestnode;           /* index of "oldest" node */
  unsigned int  i;                    /* loop counter (up to input length) */
  int j;                              /* loop counter (might get negative) */
  unsigned int  nmedian;              /* current median, outer loop counter */
  unsigned int  midpoint;             /* index of middle node in sorting order */
  unsigned int  mdnnearest;           /* checkpoint "nearest" to the median */
  unsigned int  nextnode;             /* node after an insertion point,
					 also used to find a median */
  unsigned int  prevnode;             /* node before an insertion point */
  unsigned int  rightcheckpt;         /* checkpoint 'right' of an insertion point */
  float oldvalue,newvalue;           /* old + new value of the node being replaced */
  unsigned int oldlesser,oldgreater;  /* remember the pointers of the replaced node */


   /* create nodes array */
   nodes = (struct node*)calloc(bsize, sizeof(struct node));

   /* determine checkpoint positions */
   stepchkpts = sqrt(bsize);
   /* the old form
      ncheckpts = bsize/stepchkpts;
      caused too less checkpoints at the end, leading to break the
      cost calculation */
   ncheckpts = ceil((float)bsize/(float)stepchkpts);

   /* set checkpoint nearest to the median and offset of the median to it */
   midpoint = (bsize+(bsize&1)) / 2 - 1;
#ifndef NEWCHECKPOINT
   mdnnearest = floor(midpoint / stepchkpts);
   mdnoffset = midpoint - mdnnearest * stepchkpts;
#else
   /* this becomes the median checkpoint */
   /* Hint: Bsize = 18, 32 */
   mdnnearest = ceil((float)midpoint / (float)stepchkpts);
#endif

#ifdef NEWCHECKPOINT
   /* add a checkpoint for the median if necessary */
   if (ceil((float)midpoint / (float)stepchkpts) != (float)midpoint / (float)stepchkpts)
     ncheckpts++;
#endif

   /* create checkpoints array */
   checkpts = (unsigned int*)calloc(ncheckpts,sizeof(unsigned int));

   /* create array for qsort */
   qsnodes = (struct qsnode*)calloc(bsize, sizeof(struct qsnode));

   /* init qsort array
      the nodes get their values from the input,
      the indices are only identities qi[0]=0,qi[1]=1,... */
   for(i = 0; i < bsize; i++)
     {
       qsnodes[i].value = input[i];
       qsnodes[i].index = i;
     }

   /* sort qsnodes by value and index(!) */
   qsort(qsnodes, bsize, sizeof(struct qsnode), rngmed_qsortindex);

   /* init nodes array */
   for(i = 0; i < bsize; i++)
     nodes[i].value = input[i];
   for(i = 1; i < bsize - 1; i++)
     {
       nodes[qsnodes[i-1].index].greater = qsnodes[i].index;
       nodes[qsnodes[i+1].index].lesser  = qsnodes[i].index;
     }
   nodes[qsnodes[0].index].lesser = nil; /* end marker */
   nodes[qsnodes[1].index].lesser = qsnodes[0].index;
   nodes[qsnodes[bsize-2].index].greater = qsnodes[bsize-1].index;
   nodes[qsnodes[bsize-1].index].greater = nil; /* end marker */

   /* setup checkpoints */
#ifndef NEWCHECKPOINT
   for(i = 0; i < ncheckpts; i++)
     checkpts[i] = qsnodes[i*stepchkpts].index;
#else
   /* j is the current checkpoint
      i is the stepping
      they are out of sync after a median checkpoint has been added */
   for(i = 0, j = 0; j < ncheckpts; j++, i++)
     {
       if (j == mdnnearest) {
	 checkpts[j] = qsnodes[midpoint].index;
	 if (i*stepchkpts != midpoint)
	   j++;
       }
       checkpts[j] = qsnodes[i*stepchkpts].index;
     }
#endif

   /* don't need the qsnodes anymore */
   free(qsnodes);

   /* find first median */
   nextnode = checkpts[mdnnearest];
#ifndef NEWCHECKPOINT
   for(i=0; i<mdnoffset; i++)
     nextnode = nodes[nextnode].greater;
#endif
   if(isodd)
     medians[0] = nodes[nextnode].value;
   else
     medians[0] = (nodes[nextnode].value + nodes[nodes[nextnode].greater].value) / 2.0;

   /* the "oldest" node (first in sequence) is the one with index 0 */
   oldestnode = 0;

   /* outer loop: find a median with each iteration */
   for(nmedian = 1; nmedian < length - bsize + 1; nmedian++)
     {

       /* remember value of sample to be deleted */
       oldvalue = nodes[oldestnode].value;

       /* get next value to be inserted from input */
       newvalue = input[nmedian+bsize-1];

       /** find point of insertion: **/

       /* find checkpoint greater or equal newvalue */
       /* possible optimisation: use bisectional search instaed of linear */
       for(rightcheckpt = 0; rightcheckpt < ncheckpts; rightcheckpt++)
	 if(newvalue <= nodes[checkpts[rightcheckpt]].value)
	   break;

       /* assume we are inserting at the beginning: */
       prevnode = nil;
       if (rightcheckpt == 0)
	 /* yes, we are */
	 nextnode = checkpts[0];
       else
	 {
	   /* we're beyond the first checkpoint, find the node we're
	      inserting at: */
	   nextnode = checkpts[rightcheckpt-1]; /* this also works if we found no
						   checkpoint > newvalue, as
						   then rightcheckpt == ncheckpts */

	   /* the following loop is always ran at least once, as
	      nodes[checkpts[rightcheckpt-1]].value < newvalue
	      after 'find checkpoint' loop */
	   while((nextnode != nil) && (newvalue > nodes[nextnode].value))
	     {
	       prevnode = nextnode;
	       nextnode = nodes[nextnode].greater;
	     }
	 }
       /* ok, we have:
	  - case 1: insert at beginning: prevnode == nil, nextnode ==
	  smallest node
	  - case 2: insert at end: nextnode == nil (terminated loop),
	  prevnode == last node
	  - case 3: ordinary insert: insert between prevnode and nextnode
       */

       /* insertion deletion and shifting are unnecessary if we are replacing
	  at the same pos */
       if ((oldestnode != prevnode) && (oldestnode != nextnode))
	 {

	   /* delete oldest node from list */
	   if (nodes[oldestnode].lesser == nil)
	     {
	       /* case 1: at beginning */
	       nodes[nodes[oldestnode].greater].lesser = nil;
	       /* this shouldn't be necessary, but doesn't harm */
	       checkpts[0] = nodes[oldestnode].greater;
	     }
	   else if (nodes[oldestnode].greater == nil)
	     /* case 2: at end */
	     nodes[nodes[oldestnode].lesser].greater = nil;
	   else
	     {
	       /* case 3: anywhere else */
	       nodes[nodes[oldestnode].lesser].greater = nodes[oldestnode].greater;
	       nodes[nodes[oldestnode].greater].lesser = nodes[oldestnode].lesser;
	     }
	   /* remember the old links for special case in shifting below */
	   oldgreater = nodes[oldestnode].greater;
	   oldlesser = nodes[oldestnode].lesser;


	   /* insert new node - actually we reuse the oldest one */
	   /* the value is set outside the outer "if" */
	   nodes[oldestnode].lesser = prevnode;
	   nodes[oldestnode].greater = nextnode;
	   if (prevnode != nil)
	     nodes[prevnode].greater = oldestnode;
	   if (nextnode != nil)
	     nodes[nextnode].lesser = oldestnode;


	   /* shift checkpoints */

	   /* if there is a sequence of identical values, new values are
	      inserted
	      always at the left end. Thus, the oldest value has to be the rightmost
	      of such a sequence. This requires proper init.

	      This makes shifting of the checkpoints rather easy:
	      if (oldvalue < newvalue), all checkpoints with
	      oldvalue <(=) chkptvalue < newvalue are shifted,
	      if (newvalue <= oldvalue), all checkpoints with
	      newvalue <= chkptvalue <= oldvalue are shifted.
	      <(=) means that only a checkpoint at the deleted node must be
	      shifted, no other accidently pointing to the same value.

	      Care is needed if a checkpoint to shift is the node we just deleted

	      We start at the checkpoint we know to be closest to the new node
	      satifying the above condition:
	      rightcheckpt-1 if (oldvalue < newvalue)
	      rightcheckpt othewise
	      and proceed in the direction towards the deleted node
	   */

	   if (oldvalue < newvalue)
	     {
	       /* we shift towards larger values */
	       for(j = rightcheckpt - 1; (j > 0) && (nodes[checkpts[j]].value >= oldvalue); j--)
		 if (nodes[checkpts[j]].value > oldvalue)
		   checkpts[j] = nodes[checkpts[j]].greater;
		 else if (checkpts[j] == oldestnode)
		   checkpts[j] = oldgreater;
	     }
	   else /* newvalue <= oldvalue */
	     /* we shift towards smaller values */
	     for(i = rightcheckpt; (i < ncheckpts) && (nodes[checkpts[i]].value <= oldvalue); i++)
	       if (checkpts[i] == oldestnode)
		 checkpts[i] = oldlesser;
	       else
		 checkpts[i] = nodes[checkpts[i]].lesser;

	 } /* if ((oldestnode != prevnode) && (oldestnode != nextnode)) */

       /* in any case set new value */
       nodes[oldestnode].value = newvalue;


       /* find median */
       if (newvalue == oldvalue)
	 medians[nmedian] = medians[nmedian-1];
       else
	 {
	   nextnode = checkpts[mdnnearest];
#ifndef NEWCHECKPOINT
	   for(i = 0; i < mdnoffset; i++)
	     nextnode = nodes[nextnode].greater;
#endif
	   if(isodd)
	     medians[nmedian] = nodes[nextnode].value;
	   else
	     medians[nmedian] = (nodes[nextnode].value + nodes[nodes[nextnode].greater].value) / 2.0;
	 }

       /* next oldest node */
       oldestnode = (oldestnode + 1) % bsize; /* wrap around */

     } /* for (nmedian...) */

   /* cleanup */
   free(checkpts);
   free(nodes);

}
