package com.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;

import com.model.AssignModel;
import com.model.BookModel;
import com.model.EmployeeModel;
import com.model.ServiceModel;
import com.model.SubServiceModel;
import com.model.UserModel;
import com.util.DBUtil;

public class AdminDao {

	Connection cn=null;
	int x=0;
	public int addData(ServiceModel amodel)
	{
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into mainservice(type,description,image) values(?,?,?)";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setString(1, amodel.getType());
			st.setString(2, amodel.getDescription());
			st.setString(3,amodel.getImage());
			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}

	public int addSubData(SubServiceModel ssm)
	{
		x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into subservicei(parentid,title,description,cost) values(?,?,?,?)";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setInt(1,ssm.getParentid());
			st.setString(2,ssm.getTitle());
			st.setString(3,ssm.getDescription());
			st.setString(4,ssm.getCost());
			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
	

	public int addEmployeeData(EmployeeModel ep)
	{
		x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into employee(firstname,lastname,email,phone,password,specialization,gender) values(?,?,?,?,?,?,?)";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setString(1,ep.getFirstname());
			st.setString(2,ep.getLastname());
			st.setString(3,ep.getEmail());
			st.setString(4,ep.getPhone());
			st.setString(5,ep.getPassword());
			st.setString(6,ep.getSpecialization());
			st.setString(7,ep.getGender());
			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
	public int updateUser(BookModel bmodel)
	{
		int x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="update book set status=? where bookid=?";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setString(1, "Contract Accepted");
			
			st.setInt(2, bmodel.getBookid());
			System.out.println(st);
			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
	public int insertdatetime(AssignModel amodel)
	{
		int y=0;
		cn = new DBUtil().getDatabaseConnection();
		String qry ="update assign set date=?,time=? where bookid=?";
		
		try
		{
			PreparedStatement st=cn.prepareStatement(qry);	
			
			st.setString(1,amodel.getDate());
			st.setString(2, amodel.getTime());
			st.setInt(3, amodel.getBookid());
			
		
			y=st.executeUpdate();
			cn.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		return y;
		
	}
	
	public int addAssignData(AssignModel asmodel)
	{
		int y=0;
		x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into assign(assign_id,bookid,servicemen) values(?,?,?)";
		String qry="update book set status=? where bookid=?";
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setInt(1,asmodel.getAssign_id());
			st.setInt(2,asmodel.getBookid());
			st.setString(3,asmodel.getServicemen());
			
			PreparedStatement st1=cn.prepareStatement(qry);//pre compile the query
			st1.setString(1, "Request sent by admin");
			st1.setInt(2,asmodel.getBookid());

			//execute
			x=st.executeUpdate();
			y=st1.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
	public int requestdenied(BookModel bmodel)
	{
		int x=0;
		cn = new DBUtil().getDatabaseConnection();
		String qry ="update book set status=? where bookid=?";
		
		try
		{
			PreparedStatement st=cn.prepareStatement(qry);	
			st.setString(1,"Request denied by servicemen");
			st.setInt(2, bmodel.getBookid());
			x=st.executeUpdate();
			cn.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		return x;
		
	}
	public int cancelrequest(BookModel bmodel)
	{
		int x=0;
		cn = new DBUtil().getDatabaseConnection();
		String qry ="update book set status=? where bookid=?";
		
		try
		{
			PreparedStatement st=cn.prepareStatement(qry);	
			st.setString(1,"Request canceled by admin");
			st.setInt(2, bmodel.getBookid());
			x=st.executeUpdate();
			cn.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		return x;
		
	}
	public int cancelassignment(AssignModel amodel)
	{
		int y=0;
		cn = new DBUtil().getDatabaseConnection();
		String qry ="delete from assign where bookid=?";
		
		try
		{
			PreparedStatement st=cn.prepareStatement(qry);	
			
			st.setInt(1, amodel.getBookid());
			y=st.executeUpdate();
			cn.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		return y;
		
	}
	
	public int addExtraCostData(BookModel amodel)
	{
		int x=0;
		cn = new DBUtil().getDatabaseConnection();
		String qry ="update book set status=?,extracost=? where bookid=?";
		
		try
		{
PreparedStatement st=cn.prepareStatement(qry);	
			
			st.setString(1,"Contract Closed");
			st.setString(2,amodel.getExtracost());
			st.setInt(3, amodel.getBookid());
			
		
			x=st.executeUpdate();
			cn.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		return x;
		
	}
}
