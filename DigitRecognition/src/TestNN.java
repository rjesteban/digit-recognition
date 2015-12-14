import com.jmatio.io.MatFileReader;
import Jama.*;

class TestNN{
	public static void main(String[] args){
		String file = "ex4data1.mat";

		int input_layer_size = 400;
		int hidden_layer_size = 25;
		int num_labels = 10;
		double lambda = 1;
		int num_iters = 1000;
		double alpha = 10;

		NN nn = new NN(input_layer_size, hidden_layer_size, num_labels, lambda, num_iters);
		
		MatFileReader reader = nn.load(file);
		Matrix X = new Matrix(nn.getVariable(reader, "X"));
		Matrix y = new Matrix(nn.getVariable(reader, "y"));
		

		Matrix initial_Theta1 = nn.randInitializeWeights(input_layer_size, hidden_layer_size);
		Matrix initial_Theta2 = nn.randInitializeWeights(hidden_layer_size, num_labels);


		Matrix initial_nn_params = nn.unroll(initial_Theta1, initial_Theta2);
		Matrix final_nn_params = nn.gradientDescent( X,  y,  initial_nn_params,  alpha,  num_iters,  input_layer_size,  hidden_layer_size,  num_labels,  lambda);
		
		Matrix final_theta1 = nn.reshapeTheta1( final_nn_params, hidden_layer_size, input_layer_size+1);
		Matrix final_theta2 = nn.reshapeTheta2( final_nn_params, num_labels, hidden_layer_size+1);
		

		/*
		If your implementation is correct, you should see a reported training accuracy of about 95.3% 
		(this may vary by about 1% due to the random initialization). 
		pred = predict(Theta1, Theta2, X);
		fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
		*/
		Matrix pred = nn.predict( final_theta1,  final_theta2,  X);
		double accuracy =  nn.accuracy( pred,  y);
		System.out.println("Accuracy:"+accuracy);	
	

	}	
}