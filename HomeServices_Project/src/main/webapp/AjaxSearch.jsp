    <%@ page import="java.sql.*" %>
    <%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
      
    <%  
    String name=request.getParameter("val");  
    if(name==null||name.trim().equals(""))
    {  
    	out.print("<p>Please enter name!</p>");  
    }
    else
    {  
    	try
    	{  
    		Class.forName("com.mysql.jdbc.Driver");
    		Connection con=DriverManager.getConnection("jdbc:mysql://localhost:3306/websitedata","root","");  
    		PreparedStatement ps=con.prepareStatement("select * from mainservice where type like '"+name+"%'");  
    		ResultSet rs=ps.executeQuery();  
      		if(!rs.isBeforeFirst()) 
      		{      
     			out.println("<p>No Record Found!</p>");   
    		} 
      		else
      		{  
    			out.print("<table border='1' cellpadding='2' width='100%'>");  
    			out.print("<tr style="+"color:white"+" class='searchbar'><th>Id</th><th>Service Name</th></tr>");    			while(rs.next())
    			{  
    				out.print("<tr><td  style="+"color:white"+">"+rs.getString(1)+"</td><td><a class="+"links"+" href='viewsubservice.jsp?serviceid="+rs.getString(1)+"'>"+rs.getString(2)+"</td></tr></a>");    			}  
    			out.print("</table>");  
    		}  
    		con.close();  
    	}
    	catch(Exception e)
    	{
    		out.print(e);
    	}  
    }  
    %>  