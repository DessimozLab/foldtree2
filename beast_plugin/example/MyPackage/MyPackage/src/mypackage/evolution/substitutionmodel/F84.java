package mypackage.evolution.substitutionmodel;

import beast.base.core.Citation;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.datatype.Nucleotide;
import beast.base.evolution.substitutionmodel.EigenDecomposition;
import beast.base.evolution.substitutionmodel.SubstitutionModel;
import beast.base.evolution.tree.Node;
import beast.base.inference.parameter.RealParameter;

@Description("F84 substitution model")
@Citation("Kishino, H., and M. Hasegawa. 1989. Evaluation of the maximum likelihood estimate of the evolutionary tree topologies from DNA sequence data, and the branching order in Hominoidea. Journal of Molecular Evolution 29:170-179.")
public class F84 extends SubstitutionModel.Base {
	public Input<RealParameter> kF84Input = new Input<>("kF84", "k parameter in the F84 model", Validate.REQUIRED);

	@Override
	public void initAndValidate() {
	   	super.initAndValidate();
	   	kF84Input.get().setBounds(0.0, Double.POSITIVE_INFINITY);
	}
	
	@Override
	public void getTransitionProbabilities(Node node, double fStartTime, double fEndTime, double fRate, double[] matrix) {
	       double[] freqs = frequencies.getFreqs();
	       double freqA = freqs[0];
	       double freqC = freqs[1];
	       double freqG = freqs[2];
	       double freqT = freqs[3];
	       double freqR = freqA + freqG;
	       double freqY = freqC + freqT;

	       double k = kF84Input.get().getValue();
	       double sumPiSquared = freqA*freqA + freqC*freqC + freqG*freqG + freqT*freqT;
	       double sumPiRatios = freqA*freqA/freqR + freqC*freqC/freqY + freqG*freqG/freqR + freqT*freqT/freqY;
	       double mu = (1.0 - sumPiSquared) + k*(1.0 - sumPiRatios);

	       double distance = (fStartTime - fEndTime) * fRate;
	       double expTerm = Math.exp(-mu * distance);
	       double expTermK = Math.exp(-mu * distance * (k + 1.0));

	       matrix[0]  = freqA + freqA*(1.0/freqR - 1.0)*expTerm + ((freqR - freqA)/freqR)*expTermK; //AA
	       matrix[1]  = freqC*(1.0 - expTerm); //AC
	       matrix[2]  = freqG + freqG*(1.0/freqR - 1.0)*expTerm - (freqG/freqR)*expTermK; //AG
	       matrix[3]  = freqT*(1.0 - expTerm); //AT

	       matrix[4]  = freqA*(1.0 - expTerm); //CA
	       matrix[5]  = freqC + freqC*(1.0/freqY - 1.0)*expTerm + ((freqY - freqC)/freqY)*expTermK; //CC
	       matrix[6]  = freqG*(1.0 - expTerm); //CG
	       matrix[7]  = freqT + freqT*(1.0/freqY - 1.0)*expTerm - (freqT/freqY)*expTermK; //CT

	       matrix[8]  = freqA + freqA*(1.0/freqR - 1.0)*expTerm - (freqA/freqR)*expTermK; //GA
	       matrix[9]  = freqC*(1.0 - expTerm); //GC
	       matrix[10] = freqG + freqG*(1.0/freqR - 1.0)*expTerm + ((freqR - freqG)/freqR)*expTermK; //GG
	       matrix[11] = freqT*(1.0 - expTerm); //GT

	       matrix[12] = freqA*(1.0 - expTerm); //TA
	       matrix[13] = freqC + freqC*(1.0/freqY - 1.0)*expTerm - (freqC/freqY)*expTermK; //TC
	       matrix[14] = freqG*(1.0 - expTerm); //TG
	       matrix[15] = freqT + freqT*(1.0/freqY - 1.0)*expTerm + ((freqY - freqT)/freqY)*expTermK; //TT
	}
	
	@Override
	public EigenDecomposition getEigenDecomposition(Node node) {
		return null;
	}

	@Override
	public boolean canHandleDataType(DataType dataType) {
	   if (dataType instanceof Nucleotide) {
	     return true;
	   }
	   return false;
	}	
}
