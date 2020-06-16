/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    CFWNB.java
 *    Copyright (C) 2018 Liangxiao Jiang
 */

package weka.classifiers.bayes;

import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * <!-- globalinfo-start --> Contructs Correlation-based Feature Weighted Naive
 * Bayes (CFWNB).<br/>
 * <br/>
 * For more information refer to:<br/>
 * <br/>
 * L. Jiang, L. Zhang, C. Li and J. Wu,
 * "A Correlation-based Feature Weighting Filter for Naive Bayes," in IEEE
 * Transactions on Knowledge and Data Engineering. doi:
 * 10.1109/TKDE.2018.2836440
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;@article{Jiang2018,
 *   author = "Jiang, L. and Zhang, L. and Li, C. and Wu, J.",
 *   title = "A Correlation-based Feature Weighting Filter for Naive Bayes",
 *   journal = "IEEE Transactions on Knowledge and Data Engineering",
 *   doi = "10.1109/TKDE.2018.2836440",
 *   year = "2018"
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 *
 * @author Liangxiao Jiang (ljiang@cug.edu.cn)
 * @version $Revision: 5928 $
 */
public class CFWNB extends AbstractClassifier implements TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = -4503874444306113214L;

	/** The number of each class value occurs in the dataset */
	private double[] m_ClassCounts;

	/** The number of each attribute value occurs in the dataset */
	private double[] m_AttCounts;

	/** The number of two attributes values occurs in the dataset */
	private double[][] m_AttAttCounts;

	/** The number of class and two attributes values occurs in the dataset */
	private double[][][] m_ClassAttAttCounts;

	/** The number of values for each attribute in the dataset */
	private int[] m_NumAttValues;

	/** The number of values for all attributes in the dataset */
	private int m_TotalAttValues;

	/** The number of classes in the dataset */
	private int m_NumClasses;

	/** The number of attributes including class in the dataset */
	private int m_NumAttributes;

	/** The number of instances in the dataset */
	private int m_NumInstances;

	/** The index of the class attribute in the dataset */
	private int m_ClassIndex;

	/** The starting index of each attribute in the dataset */
	private int[] m_StartAttIndex;

	/** The 2D array of the mutual information of each pair attributes */
	private double[][] m_mutualInformation;

	/** The array of mutual information between each attribute and class */
	private double[] m_Weight;

	/**
	 * Returns a string describing this classifier.
	 *
	 * @return a description of the data generator suitable for displaying in
	 *         the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Contructs Correlation-based Feature Weighted Naive Bayes (CFWNB).\n\n"
				+ "For more information refer to:\n\n" + getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed
	 * information about the technical background of this class, e.g., paper
	 * reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Jiang, L. and Zhang, L. and Li, C. and Wu, J.");
		result.setValue(Field.TITLE, "A Correlation-based Feature Weighting Filter for Naive Bayes");
		result.setValue(Field.JOURNAL, "IEEE Transactions on Knowledge and Data Engineering");
		result.setValue(Field.YEAR, "2018");
		result.setValue(Field.VOLUME, "doi: 10.1109/TKDE.2018.2836440");
		result.setValue(Field.NUMBER, "");
		result.setValue(Field.PAGES, "");
		result.setValue(Field.URL, "http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8359364&isnumber=4358933");
		result.setValue(Field.ISSN, "1041-4347");
		result.setValue(Field.MONTH, "");
		return result;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		// reset variable
		m_NumClasses = instances.numClasses();
		m_ClassIndex = instances.classIndex();
		m_NumAttributes = instances.numAttributes();
		m_NumInstances = instances.numInstances();
		m_TotalAttValues = 0;

		// allocate space for attribute reference arrays
		m_StartAttIndex = new int[m_NumAttributes];
		m_NumAttValues = new int[m_NumAttributes];

		// set the starting index of each attribute and the number of values for
		// each attribute and the total number of values for all attributes (not
		// including class).
		for (int i = 0; i < m_NumAttributes; i++) {
			m_StartAttIndex[i] = m_TotalAttValues;
			m_NumAttValues[i] = instances.attribute(i).numValues();
			m_TotalAttValues += m_NumAttValues[i];
		}

		// allocate space for counts and frequencies
		m_ClassCounts = new double[m_NumClasses];
		m_AttCounts = new double[m_TotalAttValues];
		m_AttAttCounts = new double[m_TotalAttValues][m_TotalAttValues];
		m_ClassAttAttCounts = new double[m_NumClasses][m_TotalAttValues][m_TotalAttValues];

		// Calculate the counts
		for (int k = 0; k < m_NumInstances; k++) {
			int classVal = (int) instances.instance(k).classValue();
			m_ClassCounts[classVal]++;
			int[] attIndex = new int[m_NumAttributes];
			for (int i = 0; i < m_NumAttributes; i++) {
				attIndex[i] = m_StartAttIndex[i] + (int) instances.instance(k).value(i);
				m_AttCounts[attIndex[i]]++;
			}
			for (int Att1 = 0; Att1 < m_NumAttributes; Att1++) {
				for (int Att2 = 0; Att2 < m_NumAttributes; Att2++) {
					m_AttAttCounts[attIndex[Att1]][attIndex[Att2]]++;
					m_ClassAttAttCounts[classVal][attIndex[Att1]][attIndex[Att2]]++;
				}
			}
		}

		// compute mutual information between each pair attributes (including class)
		m_mutualInformation = new double[m_NumAttributes][m_NumAttributes];
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			for (int att2 = att1 + 1; att2 < m_NumAttributes; att2++) {
				m_mutualInformation[att1][att2] = mutualInfo(att1, att2);
				m_mutualInformation[att2][att1] = m_mutualInformation[att1][att2];
			}
		}

		double ave = 0;
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			if (att1 == m_ClassIndex)
				continue;
			ave += m_mutualInformation[att1][m_ClassIndex];
		}
		ave /= (m_NumAttributes - 1);
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			if (att1 == m_ClassIndex)
				continue;
			m_mutualInformation[att1][m_ClassIndex] /= ave;
		}
		double mean = 0;
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			if (att1 == m_ClassIndex)
				continue;
			for (int att2 = 0; att2 < m_NumAttributes; att2++) {
				if (att2 == m_ClassIndex || att2 == att1)
					continue;
				mean += m_mutualInformation[att1][att2];
			}
		}
		mean /= ((m_NumAttributes - 1) * (m_NumAttributes - 2));
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			if (att1 == m_ClassIndex)
				continue;
			for (int att2 = 0; att2 < m_NumAttributes; att2++) {
				if (att2 == m_ClassIndex || att2 == att1)
					continue;
				m_mutualInformation[att1][att2] /= mean;
			}
		}

		double[] aveMutualInfo = new double[m_NumAttributes];
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			if (att1 == m_ClassIndex)
				continue;
			for (int att2 = 0; att2 < m_NumAttributes; att2++) {
				if (att2 == m_ClassIndex || att2 == att1)
					continue;
				aveMutualInfo[att1] += m_mutualInformation[att1][att2];
			}
			aveMutualInfo[att1] /= (m_NumAttributes - 2);
		}

		m_Weight = new double[m_NumAttributes];
		for (int att1 = 0; att1 < m_NumAttributes; att1++) {
			if (att1 == m_ClassIndex)
				continue;
			m_Weight[att1] = 1 / (1 + Math.exp(-(m_mutualInformation[att1][m_ClassIndex] - aveMutualInfo[att1])));
		}
	}

	/**
	 * compute the mutual information between each pair attributes (including class)
	 *
	 * @param args
	 *            att1, att2 are two attributes
	 * @return the mutual information between each pair attributes
	 */
	private double mutualInfo(int att1, int att2) throws Exception {

		double mutualInfo = 0;
		int attIndex1 = m_StartAttIndex[att1];
		int attIndex2 = m_StartAttIndex[att2];
		// 数组存储属性attr1中各个取值出现的次数
		double[] PriorsAtt1 = new double[m_NumAttValues[att1]];
		// 数组存储属性attr2中各个取值出现的次数
		double[] PriorsAtt2 = new double[m_NumAttValues[att2]];
		// 数组存储属性attr1和attr2各个取值组合出现的次数
		double[][] PriorsAtt1Att2 = new double[m_NumAttValues[att1]][m_NumAttValues[att2]];

		//计算属性attr1中每个取值出现的概率p（Ai）
		for (int i = 0; i < m_NumAttValues[att1]; i++) {
			PriorsAtt1[i] = m_AttCounts[attIndex1 + i] / m_NumInstances;
		}

		//计算属性attr2中每个取值出现的概率p（Aj）
		for (int j = 0; j < m_NumAttValues[att2]; j++) {
			PriorsAtt2[j] = m_AttCounts[attIndex2 + j] / m_NumInstances;
		}

        //计算属性attr1、attr2的联合概率p（Ai，Aj）
		for (int i = 0; i < m_NumAttValues[att1]; i++) {
			for (int j = 0; j < m_NumAttValues[att2]; j++) {
				PriorsAtt1Att2[i][j] = m_AttAttCounts[attIndex1 + i][attIndex2 + j] / m_NumInstances;
			}
		}

		//根据上面三个概率计算互信息
		for (int i = 0; i < m_NumAttValues[att1]; i++) {
			for (int j = 0; j < m_NumAttValues[att2]; j++) {
				mutualInfo += PriorsAtt1Att2[i][j] * log2(PriorsAtt1Att2[i][j], PriorsAtt1[i] * PriorsAtt2[j]);
			}
		}
		return mutualInfo;
	}

	/**
	 * compute the logarithm whose base is 2.
	 *
	 * @param args
	 *            x,y are numerator and denominator of the fraction.
	 * @return the natual logarithm of this fraction.
	 */
	private double log2(double x, double y) {

		if (x < 1e-6 || y < 1e-6)
			return 0.0;
		else
			return Math.log(x / y) / Math.log(2);
	}

	/**
	 * Calculates the class membership probabilities for the given test instance
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @exception Exception
	 *                if there is a problem generating the prediction
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {

		// Definition of local variables
		double[] probs = new double[m_NumClasses];
		// store instance's att values in an int array
		int[] attIndex = new int[m_NumAttributes];
		for (int att = 0; att < m_NumAttributes; att++) {
			attIndex[att] = m_StartAttIndex[att] + (int) instance.value(att);
		}
		// calculate probabilities for each possible class value
		for (int classVal = 0; classVal < m_NumClasses; classVal++) {
			probs[classVal] = (m_ClassCounts[classVal] + 1.0 / m_NumClasses) / (m_NumInstances + 1.0);
			for (int att = 0; att < m_NumAttributes; att++) {
				if (att == m_ClassIndex)
					continue;
				probs[classVal] *= Math.pow(
						(m_ClassAttAttCounts[classVal][attIndex[att]][attIndex[att]] + 1.0 / m_NumAttValues[att])
								/ (m_ClassCounts[classVal] + 1.0), m_Weight[att]);
			}
		}
		Utils.normalize(probs);
		return probs;
	}

	/**
	 * returns a string representation of the classifier
	 * 
	 * @return a representation of the classifier
	 */
	public String toString() {

		return "CFWNB (Correlation-based Feature Weighted Naive Bayes)";
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 5928 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param args
	 *            the options
	 */
	public static void main(String[] args) {
		runClassifier(new CFWNB(), args);
	}
}
