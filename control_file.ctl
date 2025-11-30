seqfile = sequence.phy   * path to your alignment file
     treefile = tree.nwk       * path to your tree file
      outfile = mlb            * main result file

        noisy = 2              * how much rubbish on the screen
      verbose = 0              * 1: detailed output, 0: concise
      runmode = 0              * 0: user tree (no tree search), 1 or 2: tree search

       model = 0               * 0:JC69, 1:K80, 2:F81, 3:F84, 4:HKY85, 5:T92, 6:TN93, 7:REV
       Mgene = 0               * 0:rates, 1:separate; 2:diff pi, 3:diff k, 4:all diff

   fix_kappa = 1               * 1: fix kappa, 0: estimate
       kappa = 1               * initial or fixed kappa (Ignored in JC69, but kept for syntax)

   fix_alpha = 0               * 0: estimate alpha, 1: fix alpha
       alpha = 0.5             * initial value (Must be > 0 to trigger Gamma)
       ncatG = 4               * number of gamma rate categories

     fix_rho = 0               * 0: estimate rho, 1: fix rho
         rho = 0.1             * initial value (Must be > 0 to trigger Auto-Discrete-Gamma)

      Malpha = 0               * 0: single alpha for all genes
       clock = 0               * 0: no clock, 1: global clock, 2: local clock
       nhomo = 0               * 0: estimate freq, 1: fix freq (JC69 implies equal freqs)
       getSE = 0               * 0: don't want standard errors (saves time)
 RateAncestor = 0              * (0,1,2): rates (0) or ancestral states (1 or 2)

   cleandata = 1               * 1: remove sites with ambiguity/gaps, 0: keep them