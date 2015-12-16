
import Jama.Matrix;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author rjesteban
 */
public class Testing {
    public static void main(String[] args) {
        Matrix x;
        double[][] arr = {{-1,-2,3},{-4,5,6}};
        x = new Matrix(arr);
        Matrix y;
        y = x;
        y.times(y.transpose()).print(20,20);
        
        NN nn = new NN(2,2,2,2,2);
        
        y.print(5, 3);
        
        System.out.println("after---adding 1s to col");
        
        nn.appendWithOnes(y, "col").print(5, 3);
        
        System.out.println("after---adding 1s to row");
        
        nn.appendWithOnes(y, "row").print(5, 3);
        
        
    }
}
