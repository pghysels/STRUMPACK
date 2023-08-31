C***********************************************************************
C***********************************************************************
C
C   Version:        0.4
C   Last modified:  December 27, 1994
C   Authors:        Joseph W.H. Liu
C
C   Mathematical Sciences Section, Oak Ridge National Laboratory
C
C***********************************************************************
C***********************************************************************
C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = MMDUPD
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***********************************************************************
C***********************************************************************
C*****     MMDUPD ..... MULTIPLE MINIMUM DEGREE UPDATE     *************
C***********************************************************************
C***********************************************************************
C
C     PURPOSE - THIS ROUTINE UPDATES THE DEGREES OF NODES
C        AFTER A MULTIPLE ELIMINATION STEP.
C
C     INPUT PARAMETERS -
C        EHEAD  - THE BEGINNING OF THE LIST OF ELIMINATED
C                 NODES (I.E., NEWLY FORMED ELEMENTS).
C        NEQNS  - NUMBER OF EQUATIONS.
C        (XADJ,ADJNCY) - ADJACENCY STRUCTURE.
C        DELTA  - TOLERANCE VALUE FOR MULTIPLE ELIMINATION.
C        MAXINT - MAXIMUM MACHINE REPRESENTABLE (SHORT)
C                 INTEGER.
C
C     UPDATED PARAMETERS -
C        MDEG   - NEW MINIMUM DEGREE AFTER DEGREE UPDATE.
C        (DHEAD,DFORW,DBAKW) - DEGREE DOUBLY LINKED STRUCTURE.
C        QSIZE  - SIZE OF SUPERNODE.
C        LLIST  - WORKING LINKED LIST.
C        MARKER - MARKER VECTOR FOR DEGREE UPDATE.
C        TAG    - TAG VALUE.
C
C***********************************************************************
C
      subroutine  mmdupd ( ehead, neqns, nadj, xadj, adjncy, delta,
     1                     mdeg, dhead, dforw, dbakw, qsize,
     1                     llist, marker, maxint, tag )
c
c***********************************************************************
c
#if MMDIDXSIZE == 64
         integer*8  nadj,neqns
         integer*8    adjncy(nadj), dbakw(neqns), dforw(neqns), 
     &              dhead(neqns), llist(neqns), marker(neqns), 
     &              qsize(neqns)
         integer*8    xadj(neqns+1)
         integer*8    deg   , deg0  , delta , ehead , elmnt ,
     1              enode , fnode , i     , iq2   , istop ,
     1              istrt , j     , jstop , jstrt , link  ,
     1              maxint, mdeg  , mdeg0 , mtag  , nabor ,
     1              node  , q2head, qxhead, 
     &              tag
#else
         integer*4  nadj,neqns
         integer*4    adjncy(nadj), dbakw(neqns), dforw(neqns), 
     &              dhead(neqns), llist(neqns), marker(neqns), 
     &              qsize(neqns)
         integer*4    xadj(neqns+1)
         integer*4    deg   , deg0  , delta , ehead , elmnt ,
     1              enode , fnode , i     , iq2   , istop ,
     1              istrt , j     , jstop , jstrt , link  ,
     1              maxint, mdeg  , mdeg0 , mtag  , nabor ,
     1              node  , q2head, qxhead, 
     &              tag
#endif
c
c***********************************************************************
c
         mdeg0 = mdeg + delta
         elmnt = ehead
  100    continue
c            -------------------------------------------------------
c            for each of the newly formed element, do the following.
c            (reset tag value if necessary.)
c            -------------------------------------------------------
             if  ( elmnt .le. 0 )  return
             mtag = tag + mdeg0
             if  ( mtag .lt. maxint )  go to 300
                 tag = 1
                 do  200  i = 1, neqns
                     if  ( marker(i) .lt. maxint )  marker(i) = 0
  200            continue
                 mtag = tag + mdeg0
  300        continue
c            ---------------------------------------------
c            create two linked lists from nodes associated
c            with elmnt: one with two nabors (q2head) in
c            adjacency structure, and the other with more
c            than two nabors (qxhead).  also compute deg0,
c            number of nodes in this element.
c            ---------------------------------------------
             q2head = 0
             qxhead = 0
             deg0 = 0
             link = elmnt
  400        continue
                 istrt = xadj(link)
                 istop = xadj(link+1) - 1
                 do  700  i = istrt, istop
                     enode = adjncy(i)
                     link = - enode
c$$$                     if  ( enode )  400, 800, 500
                     if  ( enode .lt. 0)  go to 400
                     if  ( enode .eq. 0)  go to 800
c
  500                continue
                     if  ( qsize(enode) .eq. 0 )  go to 700
                         deg0 = deg0 + qsize(enode)
                         marker(enode) = mtag
c                        ----------------------------------
c                        if enode requires a degree update,
c                        then do the following.
c                        ----------------------------------
                         if  ( dbakw(enode) .ne. 0 )  go to 700
c                            ---------------------------------------
c                            place either in qxhead or q2head lists.
c                            ---------------------------------------
                             if  ( dforw(enode) .eq. 2 )  go to 600
                                 llist(enode) = qxhead
                                 qxhead = enode
                                 go to 700
  600                        continue
                             llist(enode) = q2head
                             q2head = enode
  700            continue
  800        continue
c            --------------------------------------------
c            for each enode in q2 list, do the following.
c            --------------------------------------------
             enode = q2head
             iq2 = 1
  900        continue
                 if  ( enode .le. 0 )  go to 1500
                 if  ( dbakw(enode) .ne. 0 )  go to 2200
                     tag = tag + 1
                     deg = deg0
c                    ------------------------------------------
c                    identify the other adjacent element nabor.
c                    ------------------------------------------
                     istrt = xadj(enode)
                     nabor = adjncy(istrt)
                     if  ( nabor .eq. elmnt )  nabor = adjncy(istrt+1)
c                    ------------------------------------------------
c                    if nabor is uneliminated, increase degree count.
c                    ------------------------------------------------
                     link = nabor
                     if  ( dforw(nabor) .lt. 0 )  go to 1000
                         deg = deg + qsize(nabor)
                         go to 2100
 1000                continue
c                        --------------------------------------------
c                        otherwise, for each node in the 2nd element,
c                        do the following.
c                        --------------------------------------------
                         istrt = xadj(link)
                         istop = xadj(link+1) - 1
                         do  1400  i = istrt, istop
                             node = adjncy(i)
                             link = - node
                             if  ( node .eq. enode )  go to 1400
c$$$                             if  ( node )  1000, 2100, 1100
                             if  ( node .lt. 0)  go to 1000
                             if  ( node .eq. 0)  go to 2100
c
 1100                        continue
                             if  ( qsize(node) .eq. 0 )  go to 1400
                             if  ( marker(node) .ge. tag )  go to 1200
c                                -------------------------------------
c                                case when node is not yet considered.
c                                -------------------------------------
                                 marker(node) = tag
                                 deg = deg + qsize(node)
                                 go to 1400
 1200                        continue
c                            ----------------------------------------
c                            case when node is indistinguishable from
c                            enode.  merge them into a new supernode.
c                            ----------------------------------------
                             if  ( dbakw(node) .ne. 0 )  go to 1400
                             if  ( dforw(node) .ne. 2 )  go to 1300
                                 qsize(enode) = qsize(enode) +
     1                                          qsize(node)
                                 qsize(node) = 0
                                 marker(node) = maxint
                                 dforw(node) = - enode
                                 dbakw(node) = - maxint
                                 go to 1400
 1300                        continue
c                            --------------------------------------
c                            case when node is outmatched by enode.
c                            --------------------------------------
                             if  ( dbakw(node) .eq.0 )
     1                             dbakw(node) = - maxint
 1400                    continue
                         go to 2100
 1500            continue
c                ------------------------------------------------
c                for each enode in the qx list, do the following.
c                ------------------------------------------------
                 enode = qxhead
                 iq2 = 0
 1600            continue
                     if  ( enode .le. 0 )  go to 2300
                     if  ( dbakw(enode) .ne. 0 )  go to 2200
                         tag = tag + 1
                         deg = deg0
c                        ---------------------------------
c                        for each unmarked nabor of enode,
c                        do the following.
c                        ---------------------------------
                         istrt = xadj(enode)
                         istop = xadj(enode+1) - 1
                         do  2000  i = istrt, istop
                             nabor = adjncy(i)
                             if  ( nabor .eq. 0 )  go to 2100
                             if  ( marker(nabor) .ge. tag )  go to 2000
                                 marker(nabor) = tag
                                 link = nabor
c                                ------------------------------
c                                if uneliminated, include it in
c                                deg count.
c                                ------------------------------
                                 if  ( dforw(nabor) .lt. 0 )  go to 1700
                                     deg = deg + qsize(nabor)
                                     go to 2000
 1700                            continue
c                                    -------------------------------
c                                    if eliminated, include unmarked
c                                    nodes in this element into the
c                                    degree count.
c                                    -------------------------------
                                     jstrt = xadj(link)
                                     jstop = xadj(link+1) - 1
                                     do  1900  j = jstrt, jstop
                                         node = adjncy(j)
                                         link = - node
c$$$                                         if  ( node )  1700, 2000, 1800
                                         if  ( node .lt. 0)  go to 1700
                                         if  ( node .eq. 0)  go to 2000
c
 1800                                    continue
                                         if  ( marker(node) .ge. tag )
     1                                         go to 1900
                                             marker(node) = tag
                                             deg = deg + qsize(node)
 1900                                continue
 2000                    continue
 2100                continue
c                    -------------------------------------------
c                    update external degree of enode in degree
c                    structure, and mdeg (min deg) if necessary.
c                    -------------------------------------------
                     deg = deg - qsize(enode) + 1
                     fnode = dhead(deg)
                     dforw(enode) = fnode
                     dbakw(enode) = - deg
                     if  ( fnode .gt. 0 )  dbakw(fnode) = enode
                     dhead(deg) = enode
                     if  ( deg .lt. mdeg )  mdeg = deg
 2200                continue
c                    ----------------------------------
c                    get next enode in current element.
c                    ----------------------------------
                     enode = llist(enode)
                     if  ( iq2 .eq. 1 )  go to 900
                         go to 1600
 2300        continue
c            -----------------------------
c            get next element in the list.
c            -----------------------------
             tag = mtag
             elmnt = llist(elmnt)
             go to 100
c
      end
