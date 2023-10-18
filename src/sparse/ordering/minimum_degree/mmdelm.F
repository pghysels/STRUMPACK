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
C--- SPARSPAK-A (ANSI FORTRAN) RELEASE III --- NAME = MMDELM
C  (C)  UNIVERSITY OF WATERLOO   JANUARY 1984
C***********************************************************************
C***********************************************************************
C**     MMDELM ..... MULTIPLE MINIMUM DEGREE ELIMINATION     ***********
C***********************************************************************
C***********************************************************************
C
C     PURPOSE - THIS ROUTINE ELIMINATES THE NODE MDNODE OF
C        MINIMUM DEGREE FROM THE ADJACENCY STRUCTURE, WHICH
C        IS STORED IN THE QUOTIENT GRAPH FORMAT.  IT ALSO
C        TRANSFORMS THE QUOTIENT GRAPH REPRESENTATION OF THE
C        ELIMINATION GRAPH.
C
C     INPUT PARAMETERS -
C        MDNODE - NODE OF MINIMUM DEGREE.
C        MAXINT - ESTIMATE OF MAXIMUM REPRESENTABLE (SHORT)
C                 INTEGER.
C        TAG    - TAG VALUE.
C
C     UPDATED PARAMETERS -
C        (XADJ,ADJNCY) - UPDATED ADJACENCY STRUCTURE.
C        (DHEAD,DFORW,DBAKW) - DEGREE DOUBLY LINKED STRUCTURE.
C        QSIZE  - SIZE OF SUPERNODE.
C        MARKER - MARKER VECTOR.
C        LLIST  - TEMPORARY LINKED LIST OF ELIMINATED NABORS.
C
C***********************************************************************
C
      subroutine  mmdelm ( mdnode, neqns , nadj , xadj, adjncy,
     1                     dhead , dforw , dbakw, qsize, llist,
     1                     marker, maxint, tag )
c
c***********************************************************************
c
#if MMDIDXSIZE == 64
         integer*8  nadj,neqns
         integer*8    adjncy(nadj), dbakw(neqns), dforw(neqns), 
     &              dhead(neqns), llist(neqns), marker(neqns), 
     &              qsize(neqns)
         integer*8    xadj(neqns+1)
         integer*8    elmnt , i     , istop , istrt , j     ,
     1              jstop , jstrt , link  , maxint, mdnode,
     1              nabor , node  , npv   , nqnbrs, nxnode,
     1              pvnode, rlmt  , rloc  , rnode , tag   ,
     1              xqnbr
#else
         integer*4  nadj,neqns
         integer*4    adjncy(nadj), dbakw(neqns), dforw(neqns), 
     &              dhead(neqns), llist(neqns), marker(neqns), 
     &              qsize(neqns)
         integer*4    xadj(neqns+1)
         integer*4    elmnt , i     , istop , istrt , j     ,
     1              jstop , jstrt , link  , maxint, mdnode,
     1              nabor , node  , npv   , nqnbrs, nxnode,
     1              pvnode, rlmt  , rloc  , rnode , tag   ,
     1              xqnbr
#endif
c
c***********************************************************************
c
c        -----------------------------------------------
c        find reachable set and place in data structure.
c        -----------------------------------------------
         marker(mdnode) = tag
         istrt = xadj(mdnode)
         istop = xadj(mdnode+1) - 1
c        -------------------------------------------------------
c        elmnt points to the beginning of the list of eliminated
c        nabors of mdnode, and rloc gives the storage location
c        for the next reachable node.
c        -------------------------------------------------------
         elmnt = 0
         rloc = istrt
         rlmt = istop
         do  200  i = istrt, istop
             nabor = adjncy(i)
             if  ( nabor .eq. 0 )  go to 300
                 if  ( marker(nabor) .ge. tag )  go to 200
                     marker(nabor) = tag
                     if  ( dforw(nabor) .lt. 0 )  go to 100
                         adjncy(rloc) = nabor
                         rloc = rloc + 1
                         go to 200
  100                continue
                     llist(nabor) = elmnt
                     elmnt = nabor
  200    continue
  300    continue
c            -----------------------------------------------------
c            merge with reachable nodes from generalized elements.
c            -----------------------------------------------------
             if  ( elmnt .le. 0 )  go to 1000
                 adjncy(rlmt) = - elmnt
                 link = elmnt
  400            continue
                     jstrt = xadj(link)
                     jstop = xadj(link+1) - 1
                     do  800  j = jstrt, jstop
                         node = adjncy(j)
                         link = - node
c$$$                         if  ( node )  400, 900, 500
                         if  ( node .lt. 0)  go to 400
                         if  ( node .eq. 0)  go to 900
  500                    continue
                         if  ( marker(node) .ge. tag  .or.
     1                         dforw(node) .lt. 0 )  go to 800
                             marker(node) = tag
c                            ---------------------------------
c                            use storage from eliminated nodes
c                            if necessary.
c                            ---------------------------------
  600                        continue
                                 if  ( rloc .lt. rlmt )  go to 700
                                     link = - adjncy(rlmt)
                                     rloc = xadj(link)
                                     rlmt = xadj(link+1) - 1
                                     go to 600
  700                        continue
                             adjncy(rloc) = node
                             rloc = rloc + 1
  800                continue
  900            continue
                 elmnt = llist(elmnt)
                 go to 300
 1000    continue
         if  ( rloc .le. rlmt )  adjncy(rloc) = 0
c        --------------------------------------------------------
c        for each node in the reachable set, do the following ...
c        --------------------------------------------------------
         link = mdnode
 1100    continue
             istrt = xadj(link)
             istop = xadj(link+1) - 1
             do  1700  i = istrt, istop
                 rnode = adjncy(i)
                 link = - rnode
c$$$                 if  ( rnode )  1100, 1800, 1200
                 if  ( rnode .lt. 0)  go to 1100
                 if  ( rnode .eq. 0)  go to 1800
 1200            continue
c                --------------------------------------------
c                if rnode is in the degree list structure ...
c                --------------------------------------------
                 pvnode = dbakw(rnode)
                 if  ( pvnode .eq. 0  .or.
     1                 pvnode .eq. (-maxint) )  go to 1300
c                    -------------------------------------
c                    then remove rnode from the structure.
c                    -------------------------------------
                     nxnode = dforw(rnode)
                     if  ( nxnode .gt. 0 )  dbakw(nxnode) = pvnode
                     if  ( pvnode .gt. 0 )  dforw(pvnode) = nxnode
                     npv = - pvnode
                     if  ( pvnode .lt. 0 )  dhead(npv) = nxnode
 1300            continue
c                ----------------------------------------
c                purge inactive quotient nabors of rnode.
c                ----------------------------------------
                 jstrt = xadj(rnode)
                 jstop = xadj(rnode+1) - 1
                 xqnbr = jstrt
                 do  1400  j = jstrt, jstop
                     nabor = adjncy(j)
                     if  ( nabor .eq. 0 )  go to 1500
                         if  ( marker(nabor) .ge. tag )  go to 1400
                             adjncy(xqnbr) = nabor
                             xqnbr = xqnbr + 1
 1400            continue
 1500            continue
c                ----------------------------------------
c                if no active nabor after the purging ...
c                ----------------------------------------
                 nqnbrs = xqnbr - jstrt
                 if  ( nqnbrs .gt. 0 )  go to 1600
c                    -----------------------------
c                    then merge rnode with mdnode.
c                    -----------------------------
                     qsize(mdnode) = qsize(mdnode) + qsize(rnode)
                     qsize(rnode) = 0
                     marker(rnode) = maxint
                     dforw(rnode) = - mdnode
                     dbakw(rnode) = - maxint
                     go to 1700
 1600            continue
c                --------------------------------------
c                else flag rnode for degree update, and
c                add mdnode as a nabor of rnode.
c                --------------------------------------
                 dforw(rnode) = nqnbrs + 1
                 dbakw(rnode) = 0
                 adjncy(xqnbr) = mdnode
                 xqnbr = xqnbr + 1
                 if  ( xqnbr .le. jstop )  adjncy(xqnbr) = 0
c
 1700        continue
 1800    continue
         return
c
      end
